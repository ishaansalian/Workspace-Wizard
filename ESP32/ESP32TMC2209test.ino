#include <Bluepad32.h>

// Define pin connections for Motor 1
#define DIR_PIN1 15  // Direction pin for Motor 1
#define STEP_PIN1 2  // Step pin for Motor 1

// Define pin connections for Motor 2
#define DIR_PIN2 4   // Direction pin for Motor 2
#define STEP_PIN2 5  // Step pin for Motor 2

int stepsPerCommand = 400;  // Number of steps per command for movement
int stepDelay = 200;        // Delay between steps to control speed

ControllerPtr myController;  // Pointer to the connected Bluetooth controller

// Function to move motors in a specified direction
void moveMotors(bool dir1, bool dir2, int steps) {
  // Set directions for both motors
  digitalWrite(DIR_PIN1, dir1 ? HIGH : LOW);  // Motor 1 direction
  digitalWrite(DIR_PIN2, dir2 ? HIGH : LOW);  // Motor 2 direction

  // Pulse the STEP pins to move both motors the specified number of steps
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN1, HIGH);
    digitalWrite(STEP_PIN2, HIGH);
    delayMicroseconds(stepDelay);  // Delay to control speed

    digitalWrite(STEP_PIN1, LOW);
    digitalWrite(STEP_PIN2, LOW);
    delayMicroseconds(stepDelay);  // Delay to control speed
  }
  Serial.println("Done moving.");
}

// Process the input from the Bluetooth controller
void processGamepad(ControllerPtr ctl) {
  if (!ctl) return;  // If no controller is connected, do nothing

  // Get the button states
  uint16_t buttons = ctl->buttons();

  // Map buttons to motor actions
  if (buttons & 0x0004) {  // UP button -> Move forward
    Serial.println("Moving forward...");
    moveMotors(HIGH, HIGH, stepsPerCommand);
  } else if (buttons & 0x0002) {  // DOWN button -> Move backward
    Serial.println("Moving backward...");
    moveMotors(LOW, LOW, stepsPerCommand);
  } else if (buttons & 0x0008) {  // RIGHT button -> Turn right
    Serial.println("Turning right...");
    moveMotors(HIGH, LOW, stepsPerCommand);
  } else if (buttons & 0x0001) {  // LEFT button -> Turn left
    Serial.println("Turning left...");
    moveMotors(LOW, HIGH, stepsPerCommand);
  }
}

void onConnected(ControllerPtr ctl) {
  Serial.println("Controller connected.");
  myController = ctl;  // Save the connected controller pointer
}

void onDisconnected(ControllerPtr ctl) {
  Serial.println("Controller disconnected.");
  myController = nullptr;  // Clear the controller pointer
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("Bluetooth Motor Control");

  // Set up motor pins as outputs
  pinMode(DIR_PIN1, OUTPUT);
  pinMode(STEP_PIN1, OUTPUT);
  pinMode(DIR_PIN2, OUTPUT);
  pinMode(STEP_PIN2, OUTPUT);

  // Initialize Bluepad32
  BP32.setup(&onConnected, &onDisconnected);
  Serial.println("Waiting for Bluetooth controller...");
}

void loop() {
  // Fetch new data from the controller
  BP32.update();

  // Process the controller input if a controller is connected
  if (myController) {
    processGamepad(myController);
  }
}
