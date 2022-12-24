KNIFES = [
    'knife_t',
    'knife_egg',
    'knife_ghost',
    'knife_bayonet',
    'knife_butterfly',
    'knife_falchion',
    'knife_flip',
    'knife_gut',
    'knife_tactical',
    'knife_karambit',
    'knife_m9_bayonet',
    'knife_push',
    'knife_survival_bowie',
    'knife_ursus',
    'knife_gypsy_jackknife',
    'knife_stiletto',
    'knife_widowmaker']
KNIFES += [s[6:] for s in KNIFES]
KNIFES += ['knife', 'knifegg']
KNIFES += ['weapon_'+s for s in KNIFES]

PISTOLS = ['hkp2000', 'usp_silencer', 'glock', 'p250', 'fiveseven', 'tec9', 'cz75a', 'elite', 'deagle', 'revolver']
PISTOLS += ['weapon_'+s for s in PISTOLS]

SMGS = ['mp9', 'mac10', 'bizon', 'mp7', 'ump45', 'p90', 'mp5sd']
SMGS += ['weapon_'+s for s in SMGS]

RIFLES = ['famas', 'galilar', 'm4a1', 'm4a4', 'm4a1_silencer', 'ak47', 'aug', 'sg556', 'ssg08', 'awp', 'scar20', 'g3sg1']
RIFLES += ['weapon_'+s for s in RIFLES]

HEAVY = ['nova', 'm249', 'xm1014', 'mag7', 'sawedoff', 'negev']
HEAVY += ['weapon_'+s for s in HEAVY]

GRENADES = ['hegrenade', 'incgrenade', 'smokegrenade', 'flashbang', 'decoy', 'molotov']
GRENADES += ['weapon_'+s for s in GRENADES]

GEAR = ['taser']
GEAR += ['weapon_'+s for s in GEAR]
