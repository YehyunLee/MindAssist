import serial

port = '/dev/tty.MindWaveMobile'  # adjust to match your ls output
ser = serial.Serial(port, 57600)

while True:
    raw = ser.read()
    print(raw.hex(), end=' ')