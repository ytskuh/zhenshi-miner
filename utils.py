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

def generate_random_androidId():
    # Android ID通常是16个十六进制字符组成的字符串
    # 例如：e9f7d2e514805c9d
    
    # 十六进制字符包括0-9和a-f
    hex_chars = string.hexdigits.lower()[:16]  # 取0-9a-f
    
    # 生成16个随机十六进制字符
    random_hex = ''.join(random.choice(hex_chars) for _ in range(16))
    
    return random_hex

def generate_random_ip():
    # 生成随机公网IP地址
    # 避开私有IP范围:
    # 10.0.0.0 到 10.255.255.255
    # 172.16.0.0 到 172.31.255.255
    # 192.168.0.0 到 192.168.255.255
    
    # 第一段: 避开10, 127, 169, 172, 192
    first_octet_options = [i for i in range(1, 224) if i not in [10, 127, 169, 172, 192]]
    first = random.choice(first_octet_options)
    
    # 针对特殊情况处理剩余数字段
    if first == 172:
        # 避开 172.16.0.0 到 172.31.255.255
        second = random.randint(0, 15) or random.randint(32, 255)
    elif first == 192:
        # 避开 192.168.0.0
        second = random.choice([i for i in range(256) if i != 168])
    else:
        second = random.randint(0, 255)
        
    third = random.randint(0, 255)
    fourth = random.randint(1, 254)  # 避开.0和.255（通常保留）
    
    return f"{first}.{second}.{third}.{fourth}"