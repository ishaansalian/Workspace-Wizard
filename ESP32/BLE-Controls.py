import asyncio
from bleak import BleakClient

# Replace this with your ESP32's BLE address
ADDRESS = "4D9AF5DE-80E1-6702-F956-D874824235C9"

# Replace with your ESP32's BLE characteristic UUID
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Command Mappings
COMMANDS = {
    "w": "F",  # Forward
    "s": "B",  # Backward
    "a": "L",  # Left
    "d": "R",  # Right
    "q": "exit"  # Exit
}

async def connect_and_send():
    async with BleakClient(ADDRESS) as client:
        if await client.is_connected():
            print(f"Connected to {ADDRESS}")

            while True:
                # Get user input
                user_input = input("Enter command (w: Forward, s: Backward, a: Left, d: Right, q: Exit): ")

                if user_input in COMMANDS:
                    command = COMMANDS[user_input]
                    
                    # Exit condition
                    if command == "exit":
                        print("Exiting...")
                        break
                    
                    # Send the command to the ESP32 device
                    await client.write_gatt_char(CHARACTERISTIC_UUID, command.encode())
                    print(f"Sent command: {command}")
                else:
                    print("Invalid command. Please try again.")

        else:
            print("Failed to connect!")

asyncio.run(connect_and_send())
