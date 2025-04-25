import random
import base64
import hashlib
import string
import faker

def generate_random_username():
    # Generate random bytes (16 bytes in this example)
    random_bytes = random.randbytes(16)
    
    # Encode the bytes using Base64
    base64_encoded = base64.b64encode(random_bytes).decode('utf-8')
    
    # This will give you a string like ''
    return base64_encoded

def generate_random_nickname():
    # Generate a random nickname
    fake = faker.Faker()
    nickname = fake.user_name()
    
    return nickname

def username_to_password(username):
    # 固定的盐值
    tail = "SomeThingAddToUsername7527"
    
    # 拼接用户名和盐值
    salted_username = username + tail
    
    # 使用SHA-256哈希算法
    sha256 = hashlib.sha256()
    sha256.update(salted_username.encode('utf-8'))
    
    # 将哈希值转换为十六进制字符串
    hashed_password = sha256.hexdigest()
    
    return hashed_password

def generate_random_userPic():
    pic_urls = [
        "https://img3.tapimg.com/default_avatars/384aa197eceba6322c9af740d008e65e.jpg?imageMogr2/auto-orient/strip/thumbnail/!270x270r/gravity/Center/crop/270x270/format/jpg/interlace/1/quality/80",
        "https://img3.tapimg.com/default_avatars/0222bbb6df946833752c1ae27481533b.jpg?imageMogr2/auto-orient/strip/thumbnail/!270x270r/gravity/Center/crop/270x270/format/jpg/interlace/1/quality/80",
        "https://img3.tapimg.com/default_avatars/2e37ca25ddfc668fe5ef1265689e5868.jpg?imageMogr2/auto-orient/strip/thumbnail/!270x270r/gravity/Center/crop/270x270/format/jpg/interlace/1/quality/80",
        "https://img3.tapimg.com/default_avatars/8d7b8d6bfdbafca5212ba9ab29320611.jpg?imageMogr2/auto-orient/strip/thumbnail/!270x270r/gravity/Center/crop/270x270/format/jpg/interlace/1/quality/80",
        "https://img3.tapimg.com/default_avatars/b468cd2e3133f17dc68d74a28f3651d2.jpg?imageMogr2/auto-orient/strip/thumbnail/!270x270r/gravity/Center/crop/270x270/format/jpg/interlace/1/quality/80"
    ]
    # 随机选择一个图片URL
    random_pic = random.choice(pic_urls)
    return random_pic

def generate_random_androidId():
    # Android ID通常是16个十六进制字符组成的字符串
    # 例如：e9f7d2e514805c9d
    
    # 十六进制字符包括0-9和a-f
    hex_chars = string.hexdigits.lower()[:16]  # 取0-9a-f
    
    # 生成16个随机十六进制字符
    random_hex = ''.join(random.choice(hex_chars) for _ in range(16))

    return random_hex

def generate_random_ip():
    fake = faker.Faker()
    # 生成一个随机的IPv4地址
    ip = fake.ipv4()
    return ip