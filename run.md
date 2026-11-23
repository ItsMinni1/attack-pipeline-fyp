v3: # 1) Verify the chunked DSP chain works, no hardware
python txrx.py loopback "$(cat payload.txt)" --report loopback.png

# 2) See what it will look like on air
python txrx.py sendfile payload.txt --bit-dur 0.005

# 3) Full end-to-end through speaker + mic
python txrx.py sendrecv "$(cat payload.txt)" \
    --bit-dur 0.005 --sr 96000 --report decode.png

v2: pip install numpy sounddevice matplotlib

# 1) Verify the DSP chain works — no hardware involved
python txrx.py loopback "hello world" --report loopback.png

# 2) List devices and send a message
python txrx.py list
python txrx.py send "test 21.5 kHz" --sr 96000

# 3) Play and listen back in one go, get a decode + PNG report
python txrx.py sendrecv "hello 21.5 kHz" \
    --sr 96000 --report decode.png

# 4) Characterise the speaker's HF rolloff (best done with a USB mic)
python txrx.py sweep --sr 96000 --duration 4 --report sweep.png

v1: pip install numpy sounddevice

# Basic transmit of the default string
python tx.py "test 21.5 kHz"

# List devices if the wrong output is chosen
python tx.py --list

# Pick a specific output device / higher sample rate
python tx.py "hello" --sr 96000 --freq 21500

# Transmit and then listen back through the mic to check the spectrum
python tx.py "hello" --verify