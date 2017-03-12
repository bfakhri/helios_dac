# Helios Laser DAC
Digital to Analog Converter for laser projectors.

* http://pages.bitlasers.com/helios/ *

Open source, low cost USB DAC for the ISP-DB25 laser protocol. This repository consists of:
* Hardware (PCB schematic in KiCAD)
* Firmware (Atmel Studio project)
* SDK (with examples in C++ and Python)
* Extras (firmware update tool, graphics etc.)

##Developer guide

*Third party software integration*

NB: The repo might contain work in progress, use the commits marked with release tags for development.

Navigate to the folder "sdk" to find the relevant code. You can choose to use the functions documented in HeliosDacAPI.h (shared library) or HeliosDac.h (OOP class). Basic flow for using the DAC is documented in the header files just mentioned.

The driver depends on libusb. You can use the included libusb binary libraries for x86 Windows, Mac or Linux, or you can build your own. You can find the libusb source for that on their website, linked earlier in this paragraph.

If you wish to use the shared library, there are ready-made builds for x86 windows and x64 linux in the sdk folder.

Steps to compiling shared library (.so) for Linux based systems yourself:

```shell
g++ -Wall -std=c++14 -fPIC -O2 -c HeliosDacAPI.cpp
g++ -Wall -std=c++14 -fPIC -O2 -c HeliosDac.cpp
g++ -shared -o libHeliosDacAPI.so HeliosDacAPI.o HeliosDac.o libusb-1.0.so
```

*Hardware and firmware modification*

The PCB is drawn in Kicad. The firmware is written and built with Atmel Studio for the ATSAM4S2B microcontroller.
New firmware can be uploaded to the device over USB. To do this, you must reset the "GPNVM1" bit in the flash memory, which will make the microcontroller boot to the SAM-BA bootloader. You can do this by sending a special interrupt packet to the DAC. You can then access the flash using Atmel's SAM-BA software or BOSSA. There is an automatic tool for firmware updating:

* Download the firmware updating tool (only for Windows now, Mac/Linux partially done): firmwareupdater_script.zip
* Unzip, plug in the DAC and run the file "flash.bat".
* Follow the instructions on the screen (you will need to unplug and replug the device a couple of times).
