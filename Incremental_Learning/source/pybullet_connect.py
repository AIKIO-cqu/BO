import pybullet as p

# 连接第一个环境（GUI 模式）
physicsClient1 = p.connect(p.GUI)

# 连接第二个环境（DIRECT 模式）
physicsClient2 = p.connect(p.DIRECT)

print("==============================")
print(physicsClient1, physicsClient2)
print("==============================")

p.disconnect(physicsClient1)
p.disconnect(physicsClient2)