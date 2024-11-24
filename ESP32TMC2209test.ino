// Define pin connections for Motor 1
#define DIR_PIN1 15     // Direction pin for Motor 1
#define STEP_PIN1 2     // Step pin for Motor 1

// Define pin connections for Motor 2
#define DIR_PIN2 0      // Direction pin for Motor 2
#define STEP_PIN2 4     // Step pin for Motor 2

int stepsPerCommand = 400;  // Number of steps per command for movement
int stepDelay = 200;        // Delay between steps to control speed

// Function to move motors in a specified direction
void moveMotors(bool dir1, bool dir2, int steps) {
  // Set directions for both motors
  digitalWrite(DIR_PIN1, dir1 ? HIGH : LOW);  // Motor 1 direction
  digitalWrite(DIR_PIN2, dir2 ? HIGH : LOW);  // Motor 2 direction

  // Pulse the STEP pins to move both motors the specified number of steps
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN1, HIGH);
    digitalWrite(STEP_PIN2, HIGH);
    delayMicroseconds(stepDelay);   // Delay to control speed

    digitalWrite(STEP_PIN1, LOW);
    digitalWrite(STEP_PIN2, LOW);
    delayMicroseconds(stepDelay);   // Delay to control speed
  }
  Serial.println("Done moving.");
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("Serial Motor Control");

  // Set up pins as outputs
  pinMode(DIR_PIN1, OUTPUT);
  pinMode(STEP_PIN1, OUTPUT);
  pinMode(DIR_PIN2, OUTPUT);
  pinMode(STEP_PIN2, OUTPUT);

  Serial.println("Waiting for Serial commands...");
}

void loop() {
  // Check if there is any command available from Serial
  if (Serial.available() > 0) {
    String command = Serial.readString();
    command.trim();  // Remove any extra whitespace

    if (command == "front") {
      Serial.println("Moving forward...");
      moveMotors(HIGH, HIGH, stepsPerCommand);  // Both motors move forward
    } else if (command == "back") {
      Serial.println("Moving backward...");
      moveMotors(LOW, LOW, stepsPerCommand);    // Both motors move backward
    } else if (command == "right") {
      Serial.println("Turning right...");
      moveMotors(HIGH, LOW, stepsPerCommand);   // Motor 1 moves forward, Motor 2 moves backward
    } else if (command == "left") {
      Serial.println("Turning left...");
      moveMotors(LOW, HIGH, stepsPerCommand);   // Motor 1 moves backward, Motor 2 moves forward
    } else {
      Serial.println("Invalid command. Use 'front', 'back', 'right', or 'left'.");
    }
  }
}
