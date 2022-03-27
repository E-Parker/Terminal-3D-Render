# Terminal-3D-Render
A very impractical 3D rendering engine that runs in the python terminal.
do NOT try to run this program using the standard python IDE as it does not use ANSI escape codes.
If your terminal of choice does not support ANSI escape codes it will also break in the same way.

Due to the time it takes to use the print command, sometimes the screen will refresh while the program is in the
process of clearing and redrawing the screen. this causes the screen to flash rapidly and the effect worsens the
higher your refresh rate is. At 60Hz it isn't that bad, but I would still be careful running this if you suffer from
any conditions that cause light sensitivity.

NOTES:
I have not tested this on Linux or Mac OS, I run a windows PC and that's it. sorry!

if your terminal uses more colours or just different ones, replace the values in the "COLOURS" constant with those
colours. if you have them as int 0:255, simply divide the R,G,B channels by 255.

