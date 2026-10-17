# 廣告主視覺目錄

由 `training/ad_keyvisuals.py check` 產生，不要手改；要改請改程式裡的資料表再重跑。

共 **308 張**：7 個角色 × 16 個約會情境 × 2 個取景 = 224，再加 6 個優先情境每個角色多 2 張不同 seed = 84。全部 720×1280 直式，21 歲以上的角色。

Z-Image 沒有參考圖：同一角色在不同情境是「同類型的人」，不保證同一張臉，也不是 app 角色頭像那張臉。

## 角色

| 角色 | 年齡 | 性別 | 外觀（讀自 `generate_character.CHARACTERS`） | 服裝主色 | 張數 |
|---|---|---|---|---|---|
| 欣怡（xinyi） | 21 | 女 | shoulder-length wavy chestnut brown hair, almond eyes, heart-shaped face, slim build | soft pink | 44 |
| 若曦（ruoxi） | 23 | 女 | long layered hair with soft curls, dark brown eyes, oval face, athletic build | cream | 44 |
| 雨晴（yuqing） | 25 | 女 | sleek short bob haircut, sharp eyes, angular face, tall slender build | sage green | 44 |
| 思穎（mylora） | 28 | 女 | long straight black hair, dark brown eyes, oval face, mature adult features, natural healthy build | dusty blue | 44 |
| 泰宇（taeoh） | 22 | 男 | soft permed hair, cool detached eyes, refined face, toned build | charcoal | 44 |
| 賢哲（hyunjun） | 24 | 男 | tweed cut hairstyle, deep-set eyes, sharp jawline, athletic build | navy | 44 |
| 駿祺（jungi） | 25 | 男 | slicked side part hair, confident sharp eyes, angular jaw, fit build | olive | 44 |

不用的角色：mei 19 歲、wanling 18 歲、minjun 18 歲、junho 20 歲（廣告只用 21 歲以上）。

## 16 個情境

| 情境 | 主視覺（storyboard 的高光鏡頭） | 光線 | 女角服裝 | 男角服裝 | 優先 | 張數 |
|---|---|---|---|---|---|---|
| 遊樂園 | holding a cotton candy close to her or his chest and smiling, a softly lit Ferris wheel behind, the park quiet and empty of people | dusk, warm golden light on the face with the first lights of the wheel glowing behind | a {c} knit cardigan over a white tee and a denim skirt | a {c} zip-up jacket over a white tee and jeans |  | 14 |
| 看夜景 | leaning on a railing at a hillside lookout, wind in the hair, a wide sea of city lights behind | blue hour, a soft lamp on the railing lighting the face clearly, the city lights glittering behind | a {c} coat over a plain knit top, a light scarf | a {c} jacket over a plain knit sweater | ✓ | 28 |
| 貓咪咖啡廳 | sitting on a window cushion at a small cafe table, one finger raised to the lips to ask for quiet, a sleeping orange tabby cat curled on the table | warm late-afternoon window light, a cosy interior | a {c} oversized sweater and a long skirt | a {c} sweater and light trousers | ✓ | 28 |
| 旅行 | sitting on a seawall at sunset wearing a straw hat, looking back over the shoulder, the sea behind | low golden sunset light, a warm glow on the face, the sea glittering | a {c} linen shirt and a light skirt | a {c} linen shirt and light trousers | ✓ | 28 |
| 逛夜市 | holding a soft plush toy beside the face and smiling, a quiet corner at the end of a street of plain red paper lanterns with no writing | warm lantern light on the face, a deep blue night sky | a {c} hoodie and jeans | a {c} jacket over a tee and jeans |  | 14 |
| 下廚 | in a home kitchen wearing an apron, laughing and pointing at a steaming pot on the stove | warm early-evening kitchen light, a little steam in the air | a {c} apron over a plain long-sleeved top | a {c} apron over a plain long-sleeved shirt | ✓ | 28 |
| 喝咖啡 | sitting by a cafe window holding a coffee cup in both hands, looking up warmly | soft dusk window light on one side of the face, a warm interior | a {c} knit top and a long cardigan | a {c} shirt and a light jacket | ✓ | 28 |
| 水族館 | standing before a huge tank that glows blue, a few small fish far away in the water, pointing sideways at the tank with a delighted look | soft blue light from the tank on the face, the face still clearly lit | a {c} sweater and jeans | a {c} sweater and jeans |  | 14 |
| 野餐 | sitting on a picnic blanket beside a wicker basket, hugging the knees, a calm lake behind, looking at the camera | golden afternoon sun, a warm glow, gentle shade | a {c} cotton blouse and a long skirt | a {c} cotton shirt and light trousers | ✓ | 28 |
| 拍照 | holding a small vintage camera at the chest, wearing a canvas cap, in front of a plain pastel-coloured wall | good soft afternoon light, the face evenly lit | a {c} cap, a light shirt and a long skirt | a {c} cap, a light shirt and chinos |  | 14 |
| 演唱會 | holding a glow stick near the face and smiling, only blurred coloured stage lights behind, nobody else in the frame | soft coloured stage light on the face, the face clearly lit | a {c} t-shirt and a light jacket | a {c} t-shirt and a light jacket |  | 14 |
| 唱歌 | sitting on a sofa in a private karaoke room holding a microphone, laughing | dim room with soft pink and blue ambient light, the face clearly lit | a {c} knit top and jeans | a {c} shirt and jeans |  | 14 |
| 看電影 | sitting in a cinema seat with a plain paper popcorn bucket on the lap, turning with a laugh, the screen out of frame | dim theatre, soft flickering light from the side on the face | a {c} sweater and jeans | a {c} sweater and jeans |  | 14 |
| 玩遊戲 | on a living-room sofa holding a plain black game controller, grinning as if about to win, the television out of frame | evening, a soft lamp and a cool glow from the television on the side of the face | a {c} hoodie and sweatpants | a {c} hoodie and sweatpants |  | 14 |
| 逛街 | walking on a quiet shopping street holding a plain paper shopping bag against the chest, wearing a warm scarf | late-afternoon sun, soft warm light on the face | a {c} scarf, a light coat and a skirt | a {c} scarf, a light coat and trousers |  | 14 |
| 看展 | in a gallery room with abstract coloured light projected over the clothes, holding a blank postcard, a quiet look | soft gallery light with patches of violet and teal light from an installation, the face clearly lit | a {c} blouse and wide trousers | a {c} shirt and trousers |  | 14 |

取景：A＝平視置中的半身；B＝三分之四側面、中遠景多拍一點場所；C、D＝A、B 換 seed（只有優先情境有）。

## 試跑（28 張）

7 個角色 × 4 個情境（下廚、看夜景、貓咪咖啡廳、野餐），取景 A。

