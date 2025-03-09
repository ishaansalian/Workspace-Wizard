#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// UUIDs for BLE Service and Characteristic
#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// Command Strings
#define FORWARD_CMD "F"
#define BACK_CMD "B"
#define LEFT_CMD "L"
#define RIGHT_CMD "R"

// Servo Pins
#define LEFT_SERVO_PIN 41 // Adjust as needed
#define RIGHT_SERVO_PIN 42 // Adjust as needed

// OLED I2C Pins
#define OLED_SDA 18
#define OLED_SCL 17

// OLED Display Setup
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Battery Measurement
#define BATTERY_PIN 12  // ADC1 Channel 7 (IO12)
#define MAX_ADC_READING 4095  // 12-bit ADC resolution
#define REF_VOLTAGE 3.3       // Reference voltage (V)
#define FULL_BATTERY_VOLTAGE 3.3  // 3.3V corresponds to 100%

// Battery sampling settings
#define NUM_SAMPLES 10       // Number of ADC samples for averaging
#define BATTERY_UPDATE_MS 5000  // Update battery reading every 5 seconds

// Servo Objects
Servo leftServo;
Servo rightServo;

// BLE Variables
BLEServer *pServer;
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;
unsigned long lastBatteryUpdate = 0; // Track battery update time

// Function to get the averaged battery voltage
float getBatteryPercentage() {
    int totalADC = 0;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        totalADC += analogRead(BATTERY_PIN);
        delay(5);  // Short delay between samples
    }

    float avgADC = totalADC / (float)NUM_SAMPLES;
    float voltage = (avgADC / MAX_ADC_READING) * REF_VOLTAGE;
    
    // Convert voltage to percentage (simple linear mapping)
    float batteryPercentage = (voltage / FULL_BATTERY_VOLTAGE) * 100.0;
    return batteryPercentage;
}

// Function to update the OLED display
void updateDisplay() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);

    // Print "Workspace Wizard" at the top
    display.setCursor(8, 5);  
    display.println("Workspace Wizard :D");
    
    // Battery Percentage
    display.setCursor(10, 25);
    float battery = getBatteryPercentage();
    display.print("Battery: ");
    display.print((int)battery); // Display as integer percentage
    display.print("%");

    // BLE Connection Status
    display.setCursor(10, 45);
    display.print("BLE: ");
    display.print(deviceConnected ? "ON" : "OFF");

    display.display();
}

// Function to stop both servos
void stop() {
  leftServo.write(90);
  rightServo.write(90);
}

// Function to move forward for a short duration
void moveForward() {
  leftServo.write(120);
  rightServo.write(60);
  delay(1000);           // Increased duration
  stop();
}

void moveBackward() {
  leftServo.write(60);
  rightServo.write(120);
  delay(1000);
  stop();
}

// Function to turn left on the spot for a short duration
void turnLeft() {
  leftServo.write(60);
  rightServo.write(60);
  delay(1000); // Adjusted for approximately 90° turn
  stop();
}

void turnRight() {
  leftServo.write(120);
  rightServo.write(120);
  delay(1000); // Adjusted for approximately 90° turn
  stop();
}

// BLE Server Callbacks
class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        updateDisplay();
    }

    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        updateDisplay();
        BLEDevice::startAdvertising(); // Restart advertising for new connections
    }
};

// BLE Write Callback (Handles Commands)
class MyCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
        String receivedValue = pCharacteristic->getValue().c_str();

        if (receivedValue == FORWARD_CMD) {
            Serial.println("Moving Forward");
            moveForward();
        } else if (receivedValue == BACK_CMD) {
            Serial.println("Moving Backward");
            moveBackward();
        } else if (receivedValue == LEFT_CMD) {
            Serial.println("Turning Left");
            turnLeft();
        } else if (receivedValue == RIGHT_CMD) {
            Serial.println("Turning Right");
            turnRight();
        } else {
            Serial.println("Unknown command, stopping");
            stop();
        }
    }
};

// BLE Write Callback (for BLE Snippet Integration)
class MySnipCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
        std::string value = std::string(pCharacteristic->getValue().c_str());
        if (value.length() > 0) {
            Serial.print("Received: ");
            Serial.println(value.c_str());
        }
    }
};

void setup() {
    Serial.begin(115200);
    
    // Initialize I2C with custom pins
    Wire.begin(OLED_SDA, OLED_SCL);

    // Initialize OLED Display
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println("SSD1306 allocation failed");
        for (;;);
    }

    // Initialize Battery Measurement
    analogReadResolution(12);  // 12-bit ADC resolution
    updateDisplay(); // Initial display

    // Attach Servos
    leftServo.attach(LEFT_SERVO_PIN);
    rightServo.attach(RIGHT_SERVO_PIN);
    stop();

    // Initialize BLE
    BLEDevice::init("ESP32_Robot");

    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    // Create BLE Service
    BLEService *pService = pServer->createService(SERVICE_UUID);

    // Create BLE Characteristic
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_READ |
        BLECharacteristic::PROPERTY_WRITE
    );

    pCharacteristic->setCallbacks(new MyCallbacks());  // Command handling callbacks

    // Start BLE Service
    pService->start();

    // Create another Characteristic for BLE Write Snippet (Optional)
    BLECharacteristic *pSnipCharacteristic = pService->createCharacteristic(
        "12345678-1234-5678-1234-56789abcdef1",
        BLECharacteristic::PROPERTY_WRITE
    );

    pSnipCharacteristic->setCallbacks(new MySnipCallbacks()); // Additional BLE command handling

    // Start Advertising
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    pAdvertising->setMinPreferred(0x12);
    pAdvertising->start();

    Serial.println("Waiting for a client to connect...");
}

void loop() {
    // Update battery reading every BATTERY_UPDATE_MS milliseconds
    if (millis() - lastBatteryUpdate > BATTERY_UPDATE_MS) {
        lastBatteryUpdate = millis();
        updateDisplay();
    }
}
