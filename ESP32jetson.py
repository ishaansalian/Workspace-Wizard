import asyncio
from bleak import BleakClient

# ESP32 BLE details
ESP32_ADDRESS = "XX:XX:XX:XX:XX:XX"  # Replace with the ESP32's MAC address
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Command Mappings
COMMANDS = {
    "w": "F",  # Forward
    "s": "B",  # Backward
    "a": "L",  # Left
    "d": "R",  # Right
    "q": "exit" # Exit
}

async def send_command(command: str):
    async with BleakClient(ESP32_ADDRESS) as client:
        if client.is_connected:
            print(f"Connected to {ESP32_ADDRESS}")
            await client.write_gatt_char(CHARACTERISTIC_UUID, command.encode())
            print(f"Sent command: {command}")
        else:
            print("Failed to connect to ESP32")

async def main():
    print("Use WASD keys to move the robot. Press 'q' to exit.")
    while True:
        command = input("Enter command: ").strip().lower()
        if command in COMMANDS:
            if COMMANDS[command] == "exit":
                print("Exiting...")
                break
            await send_command(COMMANDS[command])
        else:
            print("Invalid command. Use W/A/S/D or Q to quit.")

if __name__ == "__main__":
    asyncio.run(main())
