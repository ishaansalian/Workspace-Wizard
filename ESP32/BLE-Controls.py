import asyncio
from bleak import BleakClient, BleakScanner
import re

# Replace this with your ESP32's BLE address
# ADDRESS = "4D9AF5DE-80E1-6702-F956-D874824235C9"
ADDRESS = ""

# Replace with your ESP32's BLE characteristic UUID
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Valid command characters
VALID_COMMANDS = {"F", "B", "L", "R", "N", "M", "O", "P"}
# Custom messages
CUSTOM_MESSAGES = {"finished", "stuck"}

def parse_command(input_str):
    """
    Parse input like '10F', 'M', or custom messages like 'finished'.
    """
    input_str = input_str.strip().lower()
    if input_str in CUSTOM_MESSAGES:
        return input_str
    match = re.fullmatch(r'(\d*)([FBLRNMOP])', input_str.upper())
    if match:
        count = match.group(1)
        command = match.group(2)
        return (count if count else "1") + command
    return None

async def find_device(name):
    devices = await BleakScanner.discover()
    for device in devices:
        if device.name and name in device.name:
            return device.address
    return None

async def connect_and_send():
    ADDRESS = await find_device("ESP32_Robot")  # Adjust to match your device's name
    if not ADDRESS:
        print("ESP32 device not found!")
        return
    async with BleakClient(ADDRESS) as client:
        if await client.is_connected():
            print(f"Connected to {ADDRESS}")

            while True:
                user_input = input("Enter command (e.g., 10F, M, O, finished, stuck, q to Exit): ").strip()
                
                if user_input.lower() == "q":
                    print("Exiting...")
                    break
                
                parsed_command = parse_command(user_input)
                
                if parsed_command:
                    await client.write_gatt_char(CHARACTERISTIC_UUID, parsed_command.encode())
                    print(f"Sent command: {parsed_command}")
                else:
                    print("Invalid command. Valid formats: '10F', 'M', 'O', 'finished', 'stuck', etc.")
        else:
            print("Failed to connect!")

asyncio.run(connect_and_send())
