# Maya Better UV Packer
**Respects unselected UV shells and treats them as obstacle during packing so as to pack around them instead of overlapping on top of them.**

<div align="center">
<img width="800" height="auto" alt="screenshot" src="https://github.com/user-attachments/assets/7e5d4b6e-9c0f-415c-8376-21e9b3ab31de" />
</div>

## Demo

<a href="https://www.youtube.com/watch?v=uSfMu122WLk"><img width="2226" height="1302" alt="Watch on YouTube" src="https://github.com/user-attachments/assets/7c678842-d2f3-48a8-8928-0ec3e4165601" /> </a>

## Installation

- Download the zip from this github
- Put the folder called `uvpacker` under `%userprofile%\documents\maya\scripts\` or under `%userprofile%\documents\maya\<version number>\scripts` where version number is 2026 if you are using maya 2026
  - An example is C:\Users\Rev Oconner\documents\maya\2026\scripts
- Restart maya and you can run with

```python
from uvpacker import show_ui
show_ui()
```

## Note
- The UV packer depends on a C++ binary to speed up calculation.
- I have built it for Windows and is available however to use it on Mac or Linux you will have to build it for yourself.
- The app uses cmake and the build process should be fairly simple, there is no dependencies at all.
- However you can use native AABB method without using the binary as a fallback if you so wish.

## Roadmap
- Add grouping feature for packing uv shells together to be treated as one unit.
