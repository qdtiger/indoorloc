Three.js **0.160.1**, distributed under the MIT license in [LICENSE](LICENSE).

`three.module.min.js` is the unmodified browser module from the [official npm package](https://www.npmjs.com/package/three/v/0.160.1). The scene embeds it so the generated HTML works offline.

`fonts/` holds the unmodified Latin variable-weight WOFF2 files of [Inter](https://github.com/rsms/inter) and [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) from Fontsource 5.3.0, both under the SIL Open Font License 1.1 ([Inter](fonts/LICENSE-Inter.txt), [JetBrains Mono](fonts/LICENSE-JetBrainsMono.txt)). The scene embeds them so exported frames do not depend on locally installed fonts.

`fonts/noto-sans-sc-subset.woff2` is a subset of the variable [Noto Sans SC](https://github.com/notofonts/noto-cjk) (SIL Open Font License 1.1, [license](fonts/LICENSE-NotoSansSC.txt)) holding only the characters used by `examples/readme_scene.js`, for the Chinese captions. After changing those captions, regenerate it from a full `NotoSansSC-VF.ttf`:

```bash
python3 -c "import sys; print(''.join(sorted({c for c in open('examples/readme_scene.js', encoding='utf-8').read() if ord(c) > 127})))" > /tmp/chars.txt
pyftsubset NotoSansSC-VF.ttf --text-file=/tmp/chars.txt --flavor=woff2 --layout-features='*' \
  --output-file=examples/readme_scene_vendor/fonts/noto-sans-sc-subset.woff2
```
