from etool import ManagerQrcode
qr_path='qr.png'#保存路径
ManagerQrcode.generate_english_qrcode(words="https://wangxianfu.top/", save_path=qr_path)# 生成不含中文的二维码