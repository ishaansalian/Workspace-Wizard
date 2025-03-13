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

BLEServer *pServer;
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;
unsigned long lastBatteryUpdate = 0;

// Servo Pins
#define LEFT_SERVO_PIN 41
#define RIGHT_SERVO_PIN 42

Servo leftServo;
Servo rightServo;

// OLED I2C Pins
#define OLED_SDA 18
#define OLED_SCL 17

// OLED Display Setup
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Battery Measurement
#define BATTERY_PIN 12
#define MAX_ADC_READING 4095
#define REF_VOLTAGE 3.3
#define NUM_SAMPLES 10
#define BATTERY_UPDATE_MS 5000

// Voltage Divider Ratio: (R1 + R2) / R2 => (1.8K + 3.3K) / 1.8K = 5.1 / 1.8
#define VOLTAGE_DIVIDER_RATIO (5.1 / 3.3)

// Define maximum battery voltage for percentage calculation
#define MAX_BATTERY_VOLTAGE 3.7

// Struct to hold battery data
struct BatteryData {
    float voltage;
    float percentage;
};

// Function to get battery voltage and percentage
BatteryData getBatteryData() {
    int totalADC = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        totalADC += analogRead(BATTERY_PIN);
        delay(5);
    }

    // Calculate the average ADC value
    float avgADC = totalADC / (float)NUM_SAMPLES;
    // Voltage at the ADC pin after voltage divider
    float voltageAtDivider = (avgADC / MAX_ADC_READING) * REF_VOLTAGE;
    // Actual battery voltage calculation
    float batteryVoltage = voltageAtDivider * VOLTAGE_DIVIDER_RATIO;
    // Calculate battery percentage
    float batteryPercentage = (batteryVoltage / MAX_BATTERY_VOLTAGE) * 100.0;
    if (batteryPercentage > 100.0) batteryPercentage = 100.0;

    // Return both values in a struct
    return {batteryVoltage, batteryPercentage};
}

void updateDisplay(const String& message = "") {
    BatteryData battery = getBatteryData();

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(8, 5);  
    display.println("Workspace Wizard :D");
    
    // Display battery percentage and voltage
    display.setCursor(10, 25);
    display.print("Battery: ");
    display.print((int)battery.percentage);
    display.print("%, ");
    display.print(battery.voltage, 1); // Display with 2 decimal places
    display.print("V");

    display.setCursor(10, 35);
    display.print("BLE: ");
    display.print(deviceConnected ? "ON" : "OFF");

    if (!message.isEmpty()) {
        display.setCursor(10, 45);
        display.print(message);
    }

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
  delay(112.5);           // Increased duration
  stop();
}

void moveBackward() {
  leftServo.write(60);
  rightServo.write(120);
  delay(112.5);
  stop();
}

// Function to turn left on the spot for a short duration
void turnLeft() {
  leftServo.write(68);
  rightServo.write(68);
  delay(1045); // Adjusted for approximately 90° turn
  stop();
}

void turnRight() {
  leftServo.write(112);
  rightServo.write(112);
  delay(1045); // Adjusted for approximately 90° turn
  stop();
}

void smallTurnLeft() {
  leftServo.write(75);
  rightServo.write(75);
  delay(170); // Adjusted for approximately 90° turn
  stop();
}

void smallTurnRight() {
  leftServo.write(105);
  rightServo.write(105);
  delay(170); // Adjusted for approximately 90° turn
  stop();
}

void DemoSpinR() {
  moveForward();
  turnRight();
  moveForward();
  turnRight();
  moveForward();
  turnRight();
  moveForward();
  turnRight();
  stop();
}

void DemoSpinL() {
  moveForward();
  turnLeft();
  moveForward();
  turnLeft();
  moveForward();
  turnLeft();
  moveForward();
  turnLeft();
  stop();
}

class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        updateDisplay();
    }
    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        updateDisplay();
        BLEDevice::startAdvertising();
    }
};

class MyCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
        String receivedValue = pCharacteristic->getValue().c_str();

        if (receivedValue == "finished" || receivedValue == "stuck") {
            Serial.println("Status: " + receivedValue);
            updateDisplay(receivedValue);
            return;
        }

        if (receivedValue.length() < 1) return;

        int repeatCount = 1;
        char commandChar;
        if (isdigit(receivedValue[0])) {
            repeatCount = receivedValue.substring(0, receivedValue.length() - 1).toInt();
            commandChar = receivedValue[receivedValue.length() - 1];
        } else {
            commandChar = receivedValue[0];
        }

        for (int i = 0; i < repeatCount; i++) {
            switch (commandChar) {
                case 'F':
                    Serial.println("Moving Forward");
                    moveForward();
                    break;
                case 'B':
                    Serial.println("Moving Backward");
                    moveBackward();
                    break;
                case 'L':
                    Serial.println("Turning Left");
                    turnLeft();
                    break;
                case 'R':
                    Serial.println("Turning Right");
                    turnRight();
                    break;
                case 'O':
                    Serial.println("Spinning Left");
                    DemoSpinL();
                    break;
                case 'P':
                    Serial.println("Spinning Right");
                    DemoSpinR();
                    break;
                case 'N':
                    Serial.println("Adjusting Left");
                    smallTurnLeft();
                    break;
                case 'M':
                    Serial.println("Adjusting Right");
                    smallTurnRight();
                    break;
                default:
                    Serial.println("Unknown command, stopping");
                    stop();
                    return;
            }
        }
    }
};

void setup() {
    Serial.begin(115200);
    Wire.begin(OLED_SDA, OLED_SCL);
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println("SSD1306 allocation failed");
        for (;;);
    }
    analogReadResolution(12);
    updateDisplay();
    leftServo.attach(LEFT_SERVO_PIN);
    rightServo.attach(RIGHT_SERVO_PIN);
    stop();
    BLEDevice::init("ESP32_Robot");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_READ |
        BLECharacteristic::PROPERTY_WRITE
    );
    pCharacteristic->setCallbacks(new MyCallbacks());
    pService->start();
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->start();
    Serial.println("Waiting for a client connection...");
}

void loop() {
    unsigned long currentMillis = millis();
    if (currentMillis - lastBatteryUpdate >= BATTERY_UPDATE_MS) {
        updateDisplay();
        lastBatteryUpdate = currentMillis;
    }
}
