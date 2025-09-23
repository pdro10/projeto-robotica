import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH

# Definição do Robô

fanuc_200ic = DHRobot([
    RevoluteDH(d=330, a=50, alpha=np.deg2rad(-90)),
    RevoluteDH(d=0,   a=300, alpha=0),
    RevoluteDH(d=0,   a=40,  alpha=np.deg2rad(-90)),
    RevoluteDH(d=320, a=0,   alpha=np.deg2rad(90)),
    RevoluteDH(d=0,   a=0,   alpha=np.deg2rad(-90)),
    RevoluteDH(d=75,  a=0,   alpha=0)
], name='Fanuc 200iC')

# Limites das Juntas
joint_limits_deg = np.array([
    [-165, 165],  # J1
    [-100, 100],  # J2
    [-133, 151],  # J3
    [-200, 200],  # J4
    [-125, 125],  # J5
    [-360, 360]   # J6
])

def dentro_dos_limites(joint_angles_deg):
    for i, angle in enumerate(joint_angles_deg):
        if not (joint_limits_deg[i, 0] <= angle <= joint_limits_deg[i, 1]):
            return False
    return True

# Trajetoria Fanuc
def gerar_trajetoria_espiral(raio_inicial_cm=10, raio_final_cm=25, altura_inicial_cm=30, altura_final_cm=50, num_voltas=2, num_pontos=150):
    theta = np.linspace(0, 2 * np.pi * num_voltas, num_pontos)
    raio = np.linspace(raio_inicial_cm, raio_final_cm, num_pontos)
    altura = np.linspace(altura_inicial_cm, altura_final_cm, num_pontos)
    x = raio * np.cos(theta) + 30  # Centro em X=30cm
    y = raio * np.sin(theta)
    z = altura
    return np.vstack((x, y, z)).T

# Cálculo da Cinemática Inversa
print("Calculando cinemática inversa para o Fanuc 200iC...")
trajetoria_cm = gerar_trajetoria_espiral()
trajetoria_mm = trajetoria_cm * 10
q0 = np.zeros(6)
solucoes_juntas_deg = []
pontos_fk_mm = []

for ponto_mm in trajetoria_mm:
    pose = SE3(*ponto_mm)
    sol = fanuc_200ic.ikine_LM(pose, q0=q0, ilimit=100)
    if sol.success and dentro_dos_limites(np.degrees(sol.q)):
        solucoes_juntas_deg.append(np.degrees(sol.q))
        pontos_fk_mm.append(fanuc_200ic.fkine(sol.q).t)
        q0 = sol.q
    else:
        solucoes_juntas_deg.append(None)
        pontos_fk_mm.append(None)

print("Cálculo finalizado.")

# Grafico da trajetoria usando Matplotlib
print("\nGerando gráfico da trajetória...")

traj_calculada_cm = np.array([p/10 for p in pontos_fk_mm if p is not None])
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajetoria_cm[:, 0], trajetoria_cm[:, 1], trajetoria_cm[:, 2],
label='Trajetória desejada (cm)', color='blue', linestyle='--')
if len(traj_calculada_cm) > 0:
    ax.plot(traj_calculada_cm[:, 0], traj_calculada_cm[:, 1], traj_calculada_cm[:, 2],
        label='Trajetória calculada (cm)', color='green')
ax.set_xlabel('X (cm)'); ax.set_ylabel('Y (cm)'); ax.set_zlabel('Z (cm)')
ax.legend(); ax.set_title("Trajetória do Robô Fanuc LR Mate 200iC")
plt.show()
