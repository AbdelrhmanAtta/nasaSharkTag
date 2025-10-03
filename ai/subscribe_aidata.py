import os
import time
from Adafruit_IO import MQTTClient
from dotenv import load_dotenv

load_dotenv()
ADAFRUIT_IO_USERNAME = os.getenv("ADAFRUIT_IO_USERNAME")
ADAFRUIT_IO_KEY = os.getenv("ADAFRUIT_IO_KEY")
FEED_ID = "tag-data" 


def connected(client):
    print("Connected to Adafruit IO! Listening for feed changes...")
    client.subscribe(FEED_ID)

def disconnected(client):
    print("Disconnected from Adafruit IO!")
    exit(1)

def message(client, feed_id, payload):
    print(f"Feed {feed_id} received new value: {payload}")

# Create MQTT client instance
client = MQTTClient(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

# Assign callback functions
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message

# Connect and block forever
client.connect()
client.loop_blocking()
