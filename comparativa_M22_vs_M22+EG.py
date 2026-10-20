#%% Librerias y paquetes
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from scipy.interpolate import interp1d
from clase_resultados import ResultadosESAR
pendiente_HvsI = 3716.3 # 1/m
ordenada_HvsI = 1297.0 # A/m

#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))

                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor

                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer SAR, tau y Hc de resultados
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    Hc = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m'])
    return SAR, tau, Hc
#%% funcion banda temperatura
def banda_temperatura(t, T, N=500, kind='linear'):
    """
    Interpola varias curvas T(t) sobre una grilla temporal común y
    calcula estadísticas punto a punto.

    Parameters
    ----------
    t : list of np.ndarray
        Lista de vectores de tiempo.
    T : list of np.ndarray
        Lista de vectores de temperatura.
    N : int, optional
        Número de puntos de la grilla común.
    kind : str, optional
        Tipo de interpolación (interp1d).

    Returns
    -------
    tt : list of np.ndarray
        Lista original de tiempos.
    TT : list of np.ndarray
        Lista original de temperaturas.
    t_common : np.ndarray
        Grilla temporal común.
    Tmin : np.ndarray
        Temperatura mínima en cada instante.
    Tmax : np.ndarray
        Temperatura máxima en cada instante.
    Tmean : np.ndarray
        Temperatura promedio en cada instante.
    """

    # intervalo temporal común
    tmin = max(tt.min() for tt in t)
    tmax = min(tt.max() for tt in t)

    t_common = np.linspace(tmin, tmax, N)

    # interpolación
    Ti = []
    for tt, TT in zip(t, T):
        f = interp1d(tt, TT, kind=kind)
        Ti.append(f(t_common))

    Ti = np.asarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%% M22 + EG (49% y 51%)

nombre_M22='M22'
ciclos_M22 = glob("M22/**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_M22 = glob("M22/**/*resultados.txt", recursive=True)
ciclos_M22.sort()
resultados_M22.sort()
conc_M22 =  25.4 #g/L 

print('Importando ciclos de', nombre_M22,'\n')
for i,p in enumerate(ciclos_M22):
    print('  ',i,p)
print('\n')
for i,res in enumerate(resultados_M22):
    print('  ',i,res)
print('-'*50)

SAR_M22, tau_M22, Hc_M22 = extraer_SAR_tau(resultados_M22)
res_M22=[]

nombre_M22EG='M22+EG'
ciclos_M22EG = glob("M22+EG/**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_M22EG = glob("M22+EG/**/*resultados.txt", recursive=True)
ciclos_M22EG.sort()
resultados_M22EG.sort()
conc_M22EG =  25.4*0.49 #g/L 

#saco campos mas bajos que no estan en M22
ciclos_M22EG=ciclos_M22EG[5:]
resultados_M22EG=resultados_M22EG[5:]
print('Importando ciclos de', nombre_M22EG,'\n')
for i,p in enumerate(ciclos_M22EG):
    print('  ',i,p)
print('\n')
for i,res in enumerate(resultados_M22EG):
    print('  ',i,res)
print('-'*50)

SAR_M22EG, tau_M22EG, Hc_M22EG = extraer_SAR_tau(resultados_M22EG)
res_M22EG=[]
#%% ploteo ciclos
Idc_values = [10.0, 12.5, 15.0]
H0=[(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in Idc_values] 
print('Idc (dA)  -->  H0 (kA/m)')
for i,e in enumerate(Idc_values):
    print(f'{e:4.1f} dA  -->  {H0[i]:3.1f} kA/m')
fig00, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')

for i,e in enumerate(ciclos_M22):
    if '100dA' in e:
        _,_,_, H_M22,M_M22,_ = lector_ciclos(ciclos_M22[i])
        print('1',os.path.basename(e))
        axs[0].plot(H_M22/1000,M_M22,'-',label=f'{SAR_M22[i]:.3uS}')
    elif '125dA' in e:
        _,_,_, H_M22,M_M22,_ = lector_ciclos(ciclos_M22[i])
        print('2',os.path.basename(e))
        axs[1].plot(H_M22/1000,M_M22,'-',label=f'{SAR_M22[i]:.3uS}')
    elif '150dA' in e:
        _,_,_, H_M22,M_M22,_ = lector_ciclos(ciclos_M22[i])
        print('3',os.path.basename(e))
        axs[2].plot(H_M22/1000,M_M22,'-',label=f'{SAR_M22[i]:.3uS}')

for i,e in enumerate(ciclos_M22EG):
    if '100dA' in e:
        _,_,_, H_M22EG,M_M22EG,_ = lector_ciclos(ciclos_M22EG[i])
        print('3',os.path.basename(e))
        axs[0].plot(H_M22EG/1000,M_M22EG,'-',label=f'{SAR_M22EG[i]:.3uS}')
    
    elif '125dA' in e:
        _,_,_, H_M22EG,M_M22EG,_ = lector_ciclos(ciclos_M22EG[i])
        print('2',os.path.basename(e))
        axs[1].plot(H_M22EG/1000,M_M22EG,'-',label=f'{SAR_M22EG[i]:.3uS}')
    
    elif '150dA' in e:
        _,_,_, H_M22EG,M_M22EG,_ = lector_ciclos(ciclos_M22EG[i])
        print('3',os.path.basename(e))
        axs[2].plot(H_M22EG/1000,M_M22EG,'-',label=f'{SAR_M22EG[i]:.3uS}')

for a in axs:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left',frameon=True,
             shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio {nombre_M22} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M22:.1f} g/L')

#%%% Ciclos todos  Normalizado por concentracion
fig00, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')

for i,e in enumerate(ciclos_M22):
    if '100dA' in e:
        _,_,_, H_M22,M_M22,_ = lector_ciclos(ciclos_M22[i])
        print('1',os.path.basename(e))
        axs[0].plot(H_M22/1000,M_M22/conc_M22,'C0-',label=nombre_M22 if i==0 else '')
    elif '125dA' in e:
        _,_,_, H_M22,M_M22,_ = lector_ciclos(ciclos_M22[i])
        print('2',os.path.basename(e))
        axs[1].plot(H_M22/1000,M_M22/conc_M22,'C0-',label=nombre_M22 if i==3 else '')
    elif '150dA' in e:
        _,_,_, H_M22,M_M22,_ = lector_ciclos(ciclos_M22[i])
        print('3',os.path.basename(e))
        axs[2].plot(H_M22/1000,M_M22/conc_M22,'C0-',label=nombre_M22 if i==6 else '')

for j,e in enumerate(ciclos_M22EG):
    if '100dA' in e:
        _,_,_, H_M22EG,M_M22EG,_ = lector_ciclos(ciclos_M22EG[j])
        print('1',os.path.basename(e))
        axs[0].plot(H_M22EG/1000,M_M22EG/conc_M22EG,'C3-',label=nombre_M22EG if j==0 else '')
    elif '125dA' in e:
        _,_,_, H_M22EG,M_M22EG,_ = lector_ciclos(ciclos_M22EG[j])
        print('2',os.path.basename(e))
        axs[1].plot(H_M22EG/1000,M_M22EG/conc_M22EG,'C3-',label=nombre_M22EG if j==3 else '')
    elif '150dA' in e:
        _,_,_, H_M22EG,M_M22EG,_ = lector_ciclos(ciclos_M22EG[j])
        print('3',os.path.basename(e))
        axs[2].plot(H_M22EG/1000,M_M22EG/conc_M22EG,'C3-',label=nombre_M22EG if j==5 else '')

for a in axs:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left',frameon=True,shadow=True,ncol=2)
plt.suptitle(f'Ciclos promedio normalizados por concentracion\n{nombre_M22} ({conc_M22:.1f} g/L)  vs {nombre_M22EG} ({conc_M22EG:.1f} g/L)\n300 kHz & [38, 47, 57] kA/m')

#%%
print('Resultados M22', '='*80,'\n')
for r in resultados_M22:
    res_M22.append(ResultadosESAR(os.path.dirname(r)))

rates_M22EG = []
for i,r in enumerate(res_M22):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_M22EG.append(rate)
    print(i,f'WRate = {rate:.1f} °C/s')

# rates_M22: 9 valores -> grupos de 3
promedios_M22 = [np.mean(rates_M22[i:i+3]) for i in range(0, 9, 3)]
stds_M22 = [np.std(rates_M22[i:i+3], ddof=1) for i in range(0, 9, 3)]
rates_M22_mean = []
for m, s in zip(promedios_M22, stds_M22):
    rates_M22_mean.append(ufloat(m,s))
    print(f'{ufloat(m,s):.1uS}')
    
#%%
print('Resultados M22+EG', '='*80,'\n')
for r in resultados_M22EG:
    res_M22EG.append(ResultadosESAR(os.path.dirname(r)))
rates_M22EG = []
for i,r in enumerate(res_M22EG):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_M22EG.append(rate)
    print(i,f'WRate = {rate:.2f} °C/s')
    
# rates_M22EG: grupos de 3, 2 y 3 valores
grupos_M22EG = [rates_M22EG[:3], rates_M22EG[3:5], rates_M22EG[5:8]]
promedios_M22EG = [np.mean(g) for g in grupos_M22EG]
stds_M22EG = [np.std(g, ddof=1) for g in grupos_M22EG]
rates_M22EG_mean = []
for m, s in zip(promedios_M22EG, stds_M22EG):
    rates_M22EG_mean.append(ufloat(m,s))
    print(f'{ufloat(m,s):.1uS}')

#%% Templogs
fig02, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')


for i,r in enumerate(res_M22):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_M22.append(rate)
    print(i,f'WRate = {rate:.2f} °C/s')
    if '100dA' in r.directorio:
        axs[0].plot(r.time,r.temperatura,'C0.-',label=f'{nombre_M22}\n{rates_M22_mean[0]:.1uS}' if i==0 else '')
    elif '125dA' in r.directorio:
        axs[1].plot(r.time,r.temperatura,'C0.-',label=f'{nombre_M22}\n{rates_M22_mean[1]:.1uS}' if i==3 else '')
    elif '150dA' in r.directorio:
        axs[2].plot(r.time,r.temperatura,'C0.-',label=f'{nombre_M22}\n{rates_M22_mean[2]:.1uS}' if i==6 else '')

for i,r in enumerate(res_M22EG):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_M22EG.append(rate)
    print(i,f'WRate = {rate:.2f} °C/s')
    if '100dA' in r.directorio:
        axs[0].plot(r.time,r.temperatura,'C3.-',label=f'{nombre_M22EG}\n{rates_M22EG_mean[0]:.1uS}' if i==0 else '')
    elif '125dA' in r.directorio:
        axs[1].plot(r.time,r.temperatura,'C3.-',label=f'{nombre_M22EG}\n{rates_M22EG_mean[1]:.1uS}' if i==3 else '')
    elif '150dA' in r.directorio:
        axs[2].plot(r.time,r.temperatura,'C3.-',label=f'{nombre_M22EG}\n{rates_M22EG_mean[2]:.1uS}' if i==5 else '')



axs[0].set_ylabel('T (°C)')
for a in axs:
    a.grid()
    a.set_xlabel('t (s)')
    a.legend(loc='lower right',frameon=True,shadow=True,ncol=2,title='Warming Rate (°C/s)')
plt.suptitle(f'Templogs\n{nombre_M22} ({conc_M22:.1f} g/L)  vs {nombre_M22EG} ({conc_M22EG:.1f} g/L)\n300 kHz & [38, 47, 57] kA/m')

################################################################################################################################

#%% ploteo comparativo de errorbars de ESAR
#%% ploteo comparativo de errorbars de ESAR
categorias = ['38 kA/m', '47 kA/m', '57 kA/m']
categorias = ['38 kA/m', '47 kA/m', '57 kA/m']
x = np.arange(len(categorias))
fig03, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)

sep = 0.12
desp = np.array([-2.5,-1.5,-0.5,0.5,1.5,2.5])*sep

# Datos agrupados por campo
parametros_M22 = [SAR_M22, tau_M22, Hc_M22]
parametros_M22EG = [SAR_M22EG, tau_M22EG, Hc_M22EG]
ejes = [ax,ax2,ax3]

# Índices de los campos en cada lista
indices_M22 = [[0,1,2], [3,4,5], [6,7,8]]
indices_M22EG = [[0,1,2], [3,4], [5,6,7]]

for a,datos,datosEG in zip(ejes,parametros_M22,parametros_M22EG):
    for i in range(3):
        # M22: tres parámetros por campo
        for j,idx in enumerate(indices_M22[i]):
            s = datos[idx]
            a.bar(x[i]+desp[j], s.n, yerr=s.s, width=sep*0.95, capsize=3, color='C0')

        # M22EG: tres barras por campo, aunque haya distinta cantidad
        # de repeticiones originales para calcular los promedios
        for j,idx in enumerate(indices_M22EG[i]):
            s = datosEG[idx]
            a.bar(x[i]+desp[j+3], s.n, yerr=s.s, width=sep*0.95, capsize=3, color='C3')

for a in ejes:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)

ax.set_title('ESAR',loc='left')
ax2.set_title(r'$\tau$',loc='left')
ax3.set_title('$H_c$',loc='left')
ax.set_ylabel('ESAR (W/g)')
ax2.set_ylabel(r'$\tau$ (ns)')
ax3.set_ylabel('$H_c$ (kA/m)')
fig03.suptitle(f'Parámetros M22 vs M22EG\n300 kHz - [38, 47, 57] kA/m')

#%% Salvo figuras
fig00.savefig('00_ciclos_M22_vs_M22EG.png',dpi=300)
fig01.savefig('01_ciclos_M22_vs_M22EG_norm.png',dpi=300)
fig02.savefig('02_templogs_M22_vs_M22EG.png',dpi=300)
fig03.savefig('03_comparativa_M22_vs_M22EG_ESAR_Tau_Hc.png',dpi=300)

# %%


# %%
