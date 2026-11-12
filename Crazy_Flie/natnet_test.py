"""
This python script tries connects to the optitrack motive software in order to 
use it, it's extremely likely that  the LOCAL_IP and MOTIVE_IP need to be 
changed.
"""

import time

from natnet import DataDescriptions, DataFrame, NatNetClient

MOTIVE_IP = "192.168.50.210"
LOCAL_IP = "192.168.50.195"

def on_data_description(desc: DataDescriptions):
    print("Received data descriptions")

def on_data_frame(frame: DataFrame):
    print(frame)
    print("\n")

client = NatNetClient(
    server_ip_address=MOTIVE_IP,
    local_ip_address=LOCAL_IP,
    use_multicast=False,
)

client.on_data_description_received_event.handlers.append(
    on_data_description
)

client.on_data_frame_received_event.handlers.append(
    on_data_frame
)

with client:
    print("Connected to NatNet")

    client.request_modeldef()
    client.run_async()

    while True:
        time.sleep(1)