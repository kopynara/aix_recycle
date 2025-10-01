import os

base_dir = "/Users/kopynara/projects/datasets/raw/train"

rename_map = {
    "TS_2.직접촬영_01.금속캔_001.철캔_1": "can_steel1",
    "TS_2.직접촬영_01.금속캔_001.철캔_2": "can_steel2",
    "TS_2.직접촬영_01.금속캔_001.철캔_3": "can_steel3",
    "TS_2.직접촬영_01.금속캔_002.알루미늄캔_1": "can_alum1",
    "TS_2.직접촬영_01.금속캔_002.알루미늄캔_2": "can_alum2",
    "TS_2.직접촬영_02.종이_001.종이_1": "paper1",
    "TS_2.직접촬영_02.종이_001.종이_2": "paper2",
    "TS_2.직접촬영_03.페트병_001.무색단일_1": "pet_clear1",
    "TS_2.직접촬영_03.페트병_001.무색단일_2": "pet_clear2",
    "TS_2.직접촬영_03.페트병_001.무색단일_3": "pet_clear3",
    "TS_2.직접촬영_03.페트병_002.유색단일_1": "pet_color1",
    "TS_2.직접촬영_03.페트병_002.유색단일_2": "pet_color2",
    "TS_2.직접촬영_03.페트병_002.유색단일_3": "pet_color3",
    "TS_2.직접촬영_04.플라스틱_001.PE_1": "plastic_pe1",
    "TS_2.직접촬영_04.플라스틱_001.PE_2": "plastic_pe2",
    "TS_2.직접촬영_04.플라스틱_002.PP_1": "plastic_pp1",
    "TS_2.직접촬영_04.플라스틱_002.PP_2": "plastic_pp2",
    "TS_2.직접촬영_04.플라스틱_002.PP_3": "plastic_pp3",
    "TS_2.직접촬영_04.플라스틱_003.PS_1": "plastic_ps1",
    "TS_2.직접촬영_04.플라스틱_003.PS_2": "plastic_ps2",
    "TS_2.직접촬영_04.플라스틱_003.PS_3": "plastic_ps3",
    "TS_2.직접촬영_05.스티로폼_001.스티로폼_1": "styrofoam1",
    "TS_2.직접촬영_05.스티로폼_001.스티로폼_2": "styrofoam2",
    "TS_2.직접촬영_06.비닐_001.비닐": "vinyl",
    "TS_2.직접촬영_07.유리병_001.갈색": "glass_brown",
    "TS_2.직접촬영_07.유리병_002.녹색": "glass_green",
    "TS_2.영상추출_07.유리병_003.투명": "glass_clear",
    "TS_2.직접촬영_08.건전지_001.건전지": "battery",
    "TS_2.직접촬영_09.형광등_001.형광등": "fluorescent_lamp"
}

for old_name, new_name in rename_map.items():
    old_path = os.path.join(base_dir, old_name)
    new_path = os.path.join(base_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"✅ {old_name} → {new_name}")
    else:
        print(f"⚠️ {old_name} 없음 (이미 변경됐을 수도 있음)")
