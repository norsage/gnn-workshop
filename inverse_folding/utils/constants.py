LETTER_TO_INDEX = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
INDEX_TO_LETTER = {v: k for k, v in LETTER_TO_INDEX.items()}

ALL_CHOTHIA_KEYS = {
    "H1": 0,
    "H2": 1,
    "H3": 2,
    "H4": 3,
    "H5": 4,
    "H6": 5,
    "H6A": 6,
    "H6B": 7,
    "H7": 8,
    "H8": 9,
    "H9": 10,
    "H10": 11,
    "H11": 12,
    "H12": 13,
    "H13": 14,
    "H14": 15,
    "H15": 16,
    "H16": 17,
    "H17": 18,
    "H18": 19,
    "H19": 20,
    "H20": 21,
    "H21": 22,
    "H22": 23,
    "H23": 24,
    "H24": 25,
    "H25": 26,
    "H26": 27,
    "H27": 28,
    "H28": 29,
    "H29": 30,
    "H30": 31,
    "H31": 32,
    "H31A": 33,
    "H31B": 34,
    "H31C": 35,
    "H31D": 36,
    "H31E": 37,
    "H31F": 38,
    "H31G": 39,
    "H31H": 40,
    "H31I": 41,
    "H31J": 42,
    "H31K": 43,
    "H32": 44,
    "H33": 45,
    "H34": 46,
    "H35": 47,
    "H36": 48,
    "H37": 49,
    "H38": 50,
    "H39": 51,
    "H40": 52,
    "H41": 53,
    "H42": 54,
    "H43": 55,
    "H44": 56,
    "H45": 57,
    "H46": 58,
    "H47": 59,
    "H48": 60,
    "H49": 61,
    "H50": 62,
    "H51": 63,
    "H52": 64,
    "H52A": 65,
    "H52B": 66,
    "H52C": 67,
    "H52D": 68,
    "H52E": 69,
    "H52F": 70,
    "H52G": 71,
    "H52H": 72,
    "H52I": 73,
    "H52J": 74,
    "H52K": 75,
    "H52L": 76,
    "H52M": 77,
    "H52N": 78,
    "H52O": 79,
    "H53": 80,
    "H54": 81,
    "H55": 82,
    "H56": 83,
    "H57": 84,
    "H58": 85,
    "H59": 86,
    "H59A": 87,
    "H60": 88,
    "H61": 89,
    "H62": 90,
    "H63": 91,
    "H64": 92,
    "H65": 93,
    "H66": 94,
    "H66A": 95,
    "H66B": 96,
    "H67": 97,
    "H68": 98,
    "H69": 99,
    "H70": 100,
    "H70A": 101,
    "H70B": 102,
    "H70C": 103,
    "H70D": 104,
    "H70E": 105,
    "H71": 106,
    "H72": 107,
    "H73": 108,
    "H73A": 109,
    "H73B": 110,
    "H73C": 111,
    "H73D": 112,
    "H73E": 113,
    "H73F": 114,
    "H73G": 115,
    "H73H": 116,
    "H74": 117,
    "H74A": 118,
    "H74B": 119,
    "H74C": 120,
    "H74D": 121,
    "H75": 122,
    "H76": 123,
    "H76A": 124,
    "H76B": 125,
    "H76C": 126,
    "H76D": 127,
    "H76E": 128,
    "H76F": 129,
    "H76G": 130,
    "H77": 131,
    "H78": 132,
    "H79": 133,
    "H80": 134,
    "H81": 135,
    "H82": 136,
    "H82A": 137,
    "H82B": 138,
    "H82C": 139,
    "H83": 140,
    "H84": 141,
    "H85": 142,
    "H86": 143,
    "H87": 144,
    "H88": 145,
    "H89": 146,
    "H90": 147,
    "H91": 148,
    "H92": 149,
    "H93": 150,
    "H94": 151,
    "H95": 152,
    "H96": 153,
    "H97": 154,
    "H98": 155,
    "H99": 156,
    "H100": 157,
    "H100A": 158,
    "H100B": 159,
    "H100C": 160,
    "H100D": 161,
    "H100E": 162,
    "H100F": 163,
    "H100G": 164,
    "H100H": 165,
    "H100I": 166,
    "H100J": 167,
    "H100K": 168,
    "H100L": 169,
    "H100M": 170,
    "H100N": 171,
    "H100O": 172,
    "H100P": 173,
    "H100Q": 174,
    "H100R": 175,
    "H100S": 176,
    "H100T": 177,
    "H100U": 178,
    "H100V": 179,
    "H100W": 180,
    "H100X": 181,
    "H100Y": 182,
    "H100Z": 183,
    "H101": 184,
    "H101A": 185,
    "H102": 186,
    "H103": 187,
    "H104": 188,
    "H105": 189,
    "H106": 190,
    "H107": 191,
    "H107A": 192,
    "H108": 193,
    "H109": 194,
    "H110": 195,
    "H111": 196,
    "H112": 197,
    "H113": 198,
    "L1": 199,
    "L2": 200,
    "L3": 201,
    "L4": 202,
    "L5": 203,
    "L6": 204,
    "L7": 205,
    "L8": 206,
    "L9": 207,
    "L10": 208,
    "L11": 209,
    "L12": 210,
    "L13": 211,
    "L14": 212,
    "L15": 213,
    "L16": 214,
    "L17": 215,
    "L18": 216,
    "L19": 217,
    "L20": 218,
    "L21": 219,
    "L22": 220,
    "L23": 221,
    "L24": 222,
    "L25": 223,
    "L26": 224,
    "L27": 225,
    "L28": 226,
    "L29": 227,
    "L30": 228,
    "L30A": 229,
    "L30B": 230,
    "L30C": 231,
    "L30D": 232,
    "L30E": 233,
    "L30F": 234,
    "L30G": 235,
    "L30H": 236,
    "L31": 237,
    "L32": 238,
    "L33": 239,
    "L34": 240,
    "L35": 241,
    "L36": 242,
    "L37": 243,
    "L38": 244,
    "L39": 245,
    "L39A": 246,
    "L40": 247,
    "L41": 248,
    "L42": 249,
    "L43": 250,
    "L44": 251,
    "L45": 252,
    "L46": 253,
    "L47": 254,
    "L48": 255,
    "L49": 256,
    "L50": 257,
    "L51": 258,
    "L52": 259,
    "L52A": 260,
    "L52B": 261,
    "L52C": 262,
    "L52D": 263,
    "L52E": 264,
    "L53": 265,
    "L54": 266,
    "L54A": 267,
    "L54B": 268,
    "L54C": 269,
    "L54D": 270,
    "L55": 271,
    "L56": 272,
    "L57": 273,
    "L58": 274,
    "L59": 275,
    "L60": 276,
    "L61": 277,
    "L62": 278,
    "L63": 279,
    "L64": 280,
    "L65": 281,
    "L66": 282,
    "L66A": 283,
    "L66B": 284,
    "L66C": 285,
    "L67": 286,
    "L68": 287,
    "L68A": 288,
    "L68B": 289,
    "L68C": 290,
    "L68D": 291,
    "L69": 292,
    "L70": 293,
    "L71": 294,
    "L72": 295,
    "L73": 296,
    "L74": 297,
    "L75": 298,
    "L76": 299,
    "L77": 300,
    "L78": 301,
    "L79": 302,
    "L80": 303,
    "L81": 304,
    "L82": 305,
    "L83": 306,
    "L84": 307,
    "L85": 308,
    "L86": 309,
    "L87": 310,
    "L88": 311,
    "L89": 312,
    "L90": 313,
    "L91": 314,
    "L92": 315,
    "L93": 316,
    "L94": 317,
    "L95": 318,
    "L95A": 319,
    "L95B": 320,
    "L95C": 321,
    "L95D": 322,
    "L95E": 323,
    "L95F": 324,
    "L95G": 325,
    "L95H": 326,
    "L95I": 327,
    "L95J": 328,
    "L96": 329,
    "L97": 330,
    "L98": 331,
    "L99": 332,
    "L100": 333,
    "L101": 334,
    "L102": 335,
    "L103": 336,
    "L104": 337,
    "L105": 338,
    "L106": 339,
    "L106A": 340,
    "L107": 341,
}
