# About
A ***very*** impractical real-time 3D rendering engine that runs in the terminal.

This is a toy project I wrote over a couple days to see what I could do with no external libaries,
just what comes with a normal Python install. It's not really usable for anything other than messing around and 
has no real useful functionality, just enough to navigate a scene, load images, and load models.   

## EPILEPSY WARNING FOR VERSIONS 0.1.3 AND BELLOW:
Due to the time it takes to use the print command, sometimes the screen will refresh while the program is in the
process of clearing and redrawing the screen. this causes the screen to flash rapidly and the effect worsens the
higher your refresh rate is. At 60Hz it isn't that bad, but I would still be careful running this if you suffer from
any light sensitivity conditions or epilepsy.

## Controls:
```
WASD   -> move 
R      -> up
F      -> down

IJKL   -> Look

ESCAPE -> Open Menu
```

## NOTES:
### Preping models for use:
if you want to use your own models with this program you have export them without normals, and with UVs.
if you're using blender, when exporting make sure to click on the "Geometry" tab in the exporting menu, check the 
"Triangulate faces" option, and un-check the "Write normals" option. It should now load correctly. 

### Keyboard input issues on Linux / MacOS:
At the time I wrote this, there were two keyboard input modules available, one which is default for Linux and MacOS,
and another that is exclusive to Windows. Because of my the arbitrary restriction of "must use only default libraries"
and that I was using Windows at the time, I chose msvcrt. 

The only place it is used is in the GetInput function. All that needs to return is an array of keycodes from all the 
keypresses that have happend since the last time the function was called. If you want to mess around with it on another 
OS you'll have to rewrite that short section of code to use whatever keyboard input module you happen to have. 
Don't forget to replace the K_ESCAPE constant.  

### Bitmap loading issues:
There are known issues with the way I wrote my bmp parser, It doesn't correctly handle padding so only images 
which have a resolution that is a power of two load properly. 

Thanks for checking this out! 

-E. Parker
