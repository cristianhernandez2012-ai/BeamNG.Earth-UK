"""build_uk_signs.py: the British Isles' traffic signs from the Department for Transport's official images, as faces on
the shared round sign body (tools/generate_kmh_speed_signs.py's sign_round_body: the pole, brackets and plate back).

    python3 tools/build_uk_signs.py

Inputs: tools/uk_signs/<diagram>.jpg (or .png), the DfT's "Traffic signs images" (https://www.gov.uk/guidance/
traffic-sign-images), named by their TSRGD 2016 diagram number. Outputs, in the mod's beamngEarth art folder:
sign_uk_<diagram>.png (1024 px, the sign scaled to the round signs' visible disc and cut out by alpha),
sign_uk_<diagram>.dae (the face: the same textured quad as the speed signs') and sign_uk.materials.json.
SHAPES: which diagrams are round (cut to a disc) and which square (a square with rounded corners).

The images: (c) Crown copyright, Department for Transport, used under the Open Government Licence v3.0
(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_kmh_speed_signs as kmh  # noqa: E402

SRC = Path(__file__).resolve().parent / 'uk_signs'
OUT = kmh.OUT
SIZE = 1024
PANEL_PX = kmh.ROUND_SIGN.plate_panel_px     # the visible disc's radius in the texture (the body's plate)
SHAPES = {'616': 'round', '610': 'round', '816': 'square'}
CORNER = 0.08                                 # a square sign's corner radius, a share of its side


def face(path, shape):
    """The 1024 px RGBA face of a DfT image: the sign scaled so its outline fills the plate's disc (a square sign
    its inscribed square), everything outside it fully transparent (the material alpha-tests)."""
    src = Image.open(path).convert('RGB')
    w, h = src.size
    side = 2 * PANEL_PX if shape == 'round' else int(2 * PANEL_PX * 0.92)
    sq = min(w, h)
    src = src.crop(((w - sq) // 2, (h - sq) // 2, (w - sq) // 2 + sq, (h - sq) // 2 + sq))
    src = src.resize((side, side), Image.LANCZOS)
    mask = Image.new('L', (side, side), 0)
    d = ImageDraw.Draw(mask)
    if shape == 'round':
        d.ellipse((1, 1, side - 2, side - 2), fill=255)
    else:
        d.rounded_rectangle((1, 1, side - 2, side - 2), radius=int(side * CORNER), fill=255)
    out = Image.new('RGBA', (SIZE, SIZE), (255, 255, 255, 0))
    off = (SIZE - side) // 2
    out.paste(src, (off, off), mask)
    out.putalpha(out.getchannel('A').point(lambda v: 255 if v >= 128 else 0))
    return out


def main():
    mats = {}
    lods = kmh.build_face_lods()
    for diagram, shape in sorted(SHAPES.items()):
        src = next((p for p in (SRC / f'{diagram}.png', SRC / f'{diagram}.jpg') if p.exists()), None)
        if src is None:
            print(f'  missing {diagram}: put the DfT image in {SRC}')
            continue
        name = f'sign_uk_{diagram}'
        png = OUT / f'{name}.png'
        face(src, shape).save(png)
        (OUT / f'{name}.dae').write_text(kmh.dae_xml(name, png.name, lods), encoding='utf-8')
        mats[name] = {
            'name': name, 'mapTo': name, 'class': 'Material',
            'Stages': [{'colorMap': f'{kmh.TEX_DIR}/{png.name}', 'diffuseColor': [1, 1, 1, 1],
                        'specular': [kmh.FACE_SPECULAR] * 3 + [1.0], 'specularPower': kmh.FACE_SPECULAR_POWER,
                        'pixelSpecular': True}, {}, {}, {}],
            'alphaTest': True, 'alphaRef': 127, 'doubleSided': False,
        }
        print(f'  {png.name}, {name}.dae')
    (OUT / 'sign_uk.materials.json').write_text(json.dumps(mats, indent=2), encoding='utf-8')
    print('  sign_uk.materials.json')


if __name__ == '__main__':
    main()
