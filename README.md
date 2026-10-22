# Assetto → Gran Turismo 1 / 2 : convertisseur de voitures

🇬🇧 [English version](README.en.md)

Mettre **n'importe quelle voiture 3D** dans **Gran Turismo** (PlayStation, version Europe
multi-langues, PAL **SCES-00984**), à la place d'une voiture du jeu : un mod **Assetto Corsa**
(dossier de la voiture : `.kn5`, `data.acd`, `skins/`) ou un modèle **glTF / GLB** (Sketchfab…),
une fiche de caractéristiques, et son propre disque du jeu. Il en sort une image disque patchée
(`.bin` + `.cue`) pour DuckStation, RetroArch, PCSX-Redux…

Gran Turismo 2 viendra ensuite (les outils existent déjà pour la MX-5 RF dans le projet
[MazdaMx5-nd-rf-ingames](https://github.com/soaresden/MazdaMx5-nd-rf-ingames), d'où vient ce
convertisseur).

**Ce dépôt ne contient aucune donnée de jeu ni aucun modèle** : chacun utilise son disque et ses
modèles. Les modèles **chiffrés** (protection du Custom Shaders Patch d'Assetto Corsa) sont refusés
avec un message : leur géométrie ne peut pas, et ne doit pas, être lue.

## Utilisation

Python 3 avec numpy et Pillow :

```bash
python convert_gt1.py --model "content/cars/ma_voiture" --specs ma_voiture.json --disc "Gran Turismo.chd" --out "GT1 ma voiture.bin"
```

- `--model` : le dossier d'une voiture d'Assetto Corsa (ou son `.kn5`), ou un `.glb` / `.gltf`.
- `--specs` : la fiche (JSON), voir [`examples/mazda_mx5_rf_nd2.json`](examples/mazda_mx5_rf_nd2.json) :
  voiture remplacée (`target`), nom (deux lignes), dimensions (mm), poids, moteur (cylindrée,
  puissance, courbe de couple en N·m), rapports, pneus, prix, peintures (nom + couleur), et pour
  un mod Assetto Corsa la peinture du dossier `skins/` à prendre (`skin`).
- `--disc` : son image de Gran Turismo PAL (`.chd`, `.cue` ou `.bin`).

Voitures remplaçables pour l'instant (entrées de CAR.DAT vérifiées) : `aminn` (Eunos Roadster),
`altsn` (Mazda Lantis), `slgan` (Subaru Legacy Touring Wagon GT-B). Le catalogue complet des
voitures du disque (`converter/gt1/cars.py`) est en cours.

## Ce qui est fait, et vérifié

- **Lecture des modèles** : `.kn5` (versions 5 et 6, données du Custom Shaders Patch ignorées,
  modèles chiffrés refusés), textures DDS / PNG, peintures du dossier `skins/` ; glTF / GLB
  (axes, échelle à la longueur réelle, avant et arrière trouvés par les feux rouges).
- **`data.acd`** d'Assetto Corsa : décodé (car.ini, engine.ini, power.lut, drivetrain.ini,
  tyres.ini…), pour préremplir la fiche et contrôler les roues.
- **Roues, blindées** :
  - trouvées par les nœuds `WHEEL_LF`… d'Assetto Corsa, ou sur la géométrie (pièces rondes,
    étroites, posées au sol, aux quatre coins) ;
  - contrôlées avant toute conversion : symétrie, même essieu, mêmes rayons, posées au sol,
    empattement et taille des pneus comparés à la fiche, rien de la carrosserie dans les pneus
    (ce qui y entre, bras ou essieux, est retiré de la coque) ; un problème arrête la conversion
    avec une explication ;
  - taille des roues du jeu réglée sur le modèle ;
  - image de la jante dessinée depuis la vraie jante (vue de l'extérieur), vérifiée ronde et
    centrée, sinon celle de la voiture d'origine est gardée ;
  - passages de roue comme sur les voitures du jeu : faces de la carrosserie au-dessus des roues
    et parois noires derrière elles dessinées avant les roues (bit 63 des faces), intérieur des
    passages en noir.
- **Modèle du jeu** : coque basse définition en anneaux (3 niveaux de détail), texture par vues
  orthographiques (côté, dessus, avant, arrière), 16 palettes de 16 couleurs, une série par
  peinture.
- **Caractéristiques** (CARINF.DAT) : nom, dimensions, poids, rapports (7 au plus), courbe de
  couple, régimes, pneus, puissance affichée, prix, noms des peintures.

Essayé dans PCSX-Redux sur trois modèles : un glTF de Sketchfab (SUV), un mod Assetto Corsa
(berline, avec `data.acd` et peintures) et un `.kn5` seul (la MX-5 RF).

## À faire

- les images du nom de la voiture et la vignette du concessionnaire (encore celles de la voiture
  remplacée) ;
- une page locale : déposer le modèle, la fiche préremplie, le disque, choisir la voiture à
  remplacer, **aperçu 3D qui tourne** du modèle puis du résultat ;
- Gran Turismo 2.

## Licence

GPL-3.0 (voir `LICENSE`). Formats du jeu : d'après les outils de pez2k (GT1DataSplitter,
GT2ModelTool) et le projet MX-5 du même auteur. Gran Turismo est une marque de Sony Interactive
Entertainment ; projet de fan, sans lien avec Sony ni Polyphony Digital.
