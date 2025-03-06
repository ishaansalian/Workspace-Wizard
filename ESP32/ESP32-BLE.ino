#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// UUIDs for the BLE service and characteristic
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// Custom command strings
#define FORWARD_CMD  "F"
#define BACK_CMD     "B"
#define LEFT_CMD     "L"
#define RIGHT_CMD    "R"

// Servo pins
#define LEFT_SERVO_PIN  41  // Adjust as needed
#define RIGHT_SERVO_PIN 42  // Adjust as needed

// Define custom I2C pins for OLED
#define OLED_SDA 18
#define OLED_SCL 17

// OLED display setup
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

Servo leftServo;
Servo rightServo;

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

// Function to update the OLED display
void updateDisplay() {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.setCursor(10, 25);
  
  if (deviceConnected) {
    display.println("Connected");
  } else {
    display.println("NOT Connected");
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
  delay(500);
  stop();
}

// Function to move backward for a short duration
void moveBackward() {
  leftServo.write(60);
  rightServo.write(120);
  delay(500);
  stop();
}

// Function to turn left on the spot for a short duration
void turnLeft() {
  leftServo.write(60);
  rightServo.write(60);
  delay(500);
  stop();
}

// Function to turn right on the spot for a short duration
void turnRight() {
  leftServo.write(120);
  rightServo.write(120);
  delay(500);
  stop();
}

// BLE Server Callbacks to track connection status
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

// Callback function when the client writes a value to the characteristic
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

void setup() {
  Serial.begin(115200);

  // Initialize I2C with custom pins
  Wire.begin(OLED_SDA, OLED_SCL);

  // Initialize OLED display
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed");
    for (;;);
  }
  updateDisplay(); // Show initial status

  // Attach servos
  leftServo.attach(LEFT_SERVO_PIN);
  rightServo.attach(RIGHT_SERVO_PIN);
  stop();

  // Create the BLE
  BLEDevice::init("ESP32_Robot");

  // Create the BLE Server
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create the BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create the BLE Characteristic
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_WRITE
                    );

  pCharacteristic->setCallbacks(new MyCallbacks());

  // Start the service
  pService->start();

  // Start advertising the BLE service
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("Waiting for a client to connect...");
}

void loop() {
  // Nothing needed here, everything is handled in callbacks
}

