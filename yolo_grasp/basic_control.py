import time

from pymodbus import FramerType
from pymodbus.client import ModbusSerialClient
from protocol.roh_registers_v1 import *

COM_PORT = 'COM4'
NODE_ID = 2

client = ModbusSerialClient(COM_PORT, baudrate = 115200)
client.connect()

if __name__ == "__main__":

    # Make a fist
    # resp = client.write_registers(ROH_FINGER_POS_TARGET0, [0], slave=NODE_ID)
    # time.sleep(2)
    # resp = client.write_registers(ROH_FINGER_POS_TARGET1, [65535, 0, 0, 0, 0, 0], slave=NODE_ID)
    # time.sleep(2)
    # time.sleep(2)
    # resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 65535, 0, 0, 0, 0], slave=NODE_ID)
    # time.sleep(2)
    # time.sleep(2)
    # resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 15570, 0, 0, 0], slave=NODE_ID)
    # time.sleep(2)
    # time.sleep(2)
    # resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 22086, 0, 0], slave=NODE_ID)
    # time.sleep(2)
    # resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 0, 65535, 0], slave=NODE_ID)
    # time.sleep(2)
    # resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 0, 0, 65535], slave=NODE_ID)
    resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 0, 65535, 0], slave=NODE_ID)
    time.sleep(2)
    resp = client.write_registers(ROH_FINGER_POS_TARGET1, [13758, 13758, 13758, 13758, 65535, 12311], slave = NODE_ID)
    time.sleep(10)

    # Open
    # resp = client.write_registers(ROH_FINGER_POS_TARGET0, [0], slave = NODE_ID)
    # time.sleep(2)
    resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 0, 0, 0], slave = NODE_ID)
    time.sleep(2)

    # Write finger angle, the value written is actual value * 100
    real_angle = 15.05
    target_angle = round(real_angle * 100)

    if (target_angle < 0):
        target_angle += 65536

    resp = client.write_registers(ROH_FINGER_ANGLE_TARGET0, [target_angle], slave = NODE_ID)
    time.sleep(2)

    # Read the current finger angle, the actual value is output value / 100
    resp = client.read_holding_registers(ROH_FINGER_ANGLE0, count = 1, slave = NODE_ID)
    current_angle = resp.registers

    if (current_angle > [32767]):
        current_angle -= 65536

    # current_angle = current_angle / 100.0
    #
    # print("Current finger angle：", current_angle)

    cur_current_angle = current_angle[0]
    cur_current_angle  = cur_current_angle  / 100.0

    print("Current finger angle：", cur_current_angle )
    

def grasp():
    resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 0, 65535, 0], slave=NODE_ID)
    time.sleep(2)
    resp = client.write_registers(ROH_FINGER_POS_TARGET1, [30000, 30000, 30000, 30000, 65535, 12311], slave = NODE_ID)
    time.sleep(3)

    # Open
    # resp = client.write_registers(ROH_FINGER_POS_TARGET0, [0], slave = NODE_ID)
    # time.sleep(2)
    
    
def open_grasp():
    resp = client.write_registers(ROH_FINGER_POS_TARGET1, [0, 0, 0, 0, 0, 0], slave = NODE_ID)
