# Assetto → Gran Turismo 1 / 2: car converter

🇫🇷 [Version française](README.md)

Put **any 3D car** into **Gran Turismo** (PlayStation, European multi-language release, PAL
**SCES-00984**), in place of one of the game's cars: an **Assetto Corsa** mod (the car's folder:
`.kn5`, `data.acd`, `skins/`) or a **glTF / GLB** model (Sketchfab…), a specification file, and
your own disc of the game. Out comes a patched disc image (`.bin` + `.cue`) for DuckStation,
RetroArch, PCSX-Redux…

Gran Turismo 2 comes next (the tools already exist for the MX-5 RF in the
[MazdaMx5-nd-rf-ingames](https://github.com/soaresden/MazdaMx5-nd-rf-ingames) project, where this
converter comes from).

**This repository holds no game data and no model**: everyone uses their own disc and models.
**Encrypted** models (the protection of Assetto Corsa's Custom Shaders Patch) are refused with a
message: their geometry cannot, and must not, be read.

## Usage

Python 3 with numpy and Pillow:

```bash
python convert_gt1.py --model "content/cars/my_car" --specs my_car.json --disc "Gran Turismo.chd" --out "GT1 my car.bin"
```

- `--model`: an Assetto Corsa car folder (or its `.kn5`), or a `.glb` / `.gltf`.
- `--specs`: the specification file (JSON), see [`examples/mazda_mx5_rf_nd2.json`](examples/mazda_mx5_rf_nd2.json):
  the car replaced (`target`), name (two lines), dimensions (mm), weight, engine (displacement,
  power, torque curve in N·m), gear ratios, tyres, price, paints (name + colour), and for an
  Assetto Corsa mod the paint of the `skins/` folder to use (`skin`); engine (`cylinders`,
  `valvetrain`, `aspiration`) and `suspension` for the SPEC page; `short_name` for the race
  screens; `emblem` (a PNG of the logo), `info` (the INFO page's text, one per language or a
  single one) for the menus.
- `--disc`: your Gran Turismo PAL image (`.chd`, `.cue` or `.bin`).

Cars that can be replaced for now (CAR.DAT entries checked): `aminn` (Eunos Roadster), `altsn`
(Mazda Lantis), `slgan` (Subaru Legacy Touring Wagon GT-B). The full catalogue of the disc's cars
(`converter/gt1/cars.py`) is in progress.

## Done, and checked

- **Reading the models**: `.kn5` (versions 5 and 6, Custom Shaders Patch data skipped, encrypted
  models refused), DDS / PNG textures, paints of the `skins/` folder; glTF / GLB (axes, scaled to
  the real length, front and rear found from the red lamps).
- Assetto Corsa's **`data.acd`**: decoded (car.ini, engine.ini, power.lut, drivetrain.ini,
  tyres.ini…), to fill the specification file and check the wheels.
- **Wheels, made robust**:
  - found from Assetto Corsa's `WHEEL_LF`… nodes, or on the geometry (round, narrow pieces
    standing on the ground at the four corners);
  - checked before anything is converted: symmetry, same axle, same radii, on the ground,
    wheelbase and tyre size against the specification file, nothing of the body inside the tyres
    (what goes into them, arms or axles, is taken out of the shell); a problem stops the
    conversion with an explanation;
  - the game's wheel size set from the model;
  - the rim picture drawn from the real rim (seen from outside), checked round and centred, else
    the original car's is kept;
  - wheel arches as on the game's cars: the body faces over the wheels and black walls behind
    them drawn before the wheels (bit 63 of the faces), inside of the arches in black.
- **The game's model**: a low-poly shell of rings (3 levels of detail), texture from orthographic
  views (side, top, front, rear), 16 palettes of 16 colours, one set per paint.
- **Specifications** (CARINF.DAT): name, dimensions, weight, gear ratios (7 at most), torque curve,
  revs, tyres, displayed power, price, paint names, and for the SPEC page the engine type
  (cylinders, valvetrain, turbo) and the front / rear suspensions.
- **Menus** (MENU_IMG.ARC, the 6 languages): the name picture (the make's emblem, name, version,
  in the game's style), the dealer tile (a picture of the converted model, emblem, name, version,
  price with each language's separator) and the INFO page (title and history); the short name
  of the race screens (SYSTEM.DAT).
- **Several cars on one disc**: `generate(..., files=...)` adds a car to the files another one
  already changed.

Tried in PCSX-Redux on three models: a glTF from Sketchfab (an SUV), an Assetto Corsa mod (a
saloon, with `data.acd` and paints) and a lone `.kn5` (the MX-5 RF).

## To do

- a local page: drop the model, the specification file filled in, the disc, choose the car to
  replace, **rotating 3D preview** of the model and then of the result;
- Gran Turismo 2.

## Licence

GPL-3.0 (see `LICENSE`). Game formats: after pez2k's tools (GT1DataSplitter, GT2ModelTool) and the
same author's MX-5 project. Gran Turismo is a trademark of Sony Interactive Entertainment; fan
project, not affiliated with Sony or Polyphony Digital.
