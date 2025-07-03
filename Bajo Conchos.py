# PMP CODE 
#Scenario 2 and 3  internal water market individual code for Bajo Conchos using Evapotranspiration (FAO EQ.56 AND IRRIGATION METHODS)
#%% Library import
import os
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
#%% Reported Data by CONAGUA for 2 Irrigatión Districts 
region = {"BConchos": {"volumen_mm3": 79206.52, "lamina_reportada": 2.46}} # en m 
volumen_hm3_distritos = {distrito: region[distrito]["volumen_mm3"] / 1_000 for distrito in region} 
# Delicias_005 Data
data = {
    'BConchos': {
        'Avena Forrajera': {'price':6113 , 'yield': 31, 'cost': 23837 , 'land':427 , 'Evapotranspiration':16.2,'IRMETHOD':'Tradicional','METHODEF':.6, 'Ky':.7},
        'Rye Grass': {'price': 906 , 'yield': 75, 'cost': 11733 , 'land':190 , 'Evapotranspiration':38.3,'IRMETHOD':'tradicional','METHODEF':.6, 'Ky':.8},
        'Algodon': {'price': 29680 , 'yield': 4.8, 'cost':47071 , 'land':106 , 'Evapotranspiration':80.80,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':.85},
        'Sorgo': {'price': 680, 'yield': 78, 'cost': 29616 , 'land': 247, 'Evapotranspiration':54.90,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':.9},
        'Alfalfa': {'price':2266 , 'yield': 84, 'cost': 32364, 'land': 1531 , 'Evapotranspiration':100.17,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':1.1},
        'NuezdeNogal': {'price':72522 , 'yield': 2.5, 'cost': 94148, 'land': 777 , 'Evapotranspiration':116.86,'IRMETHOD':'Asperción','METHODEF':.75, 'Ky':1.2}}}
#%% Initial Equations 
#Revenues for Delicias 
def calculate_revenues(region_data):
    revenues = {}
    for crop, info in region_data.items(): revenues[crop] = info['price'] * info['yield']
    return revenues
#Applied Water (Irrigation Depth)
def calculate_appliedwater(region_data):
    applied_water_m = {}
    for crop, info in region_data.items():
        lamina_cm = info['Evapotranspiration'] / info['METHODEF']  # cálculo en cm
        lamina_m = lamina_cm / 100  # conversión a metros
        applied_water_m[crop] = round(lamina_m, 4)
    return applied_water_m
#Area m2 
def area_original_m2(region_data):
    land_m2 = {crop: info['land'] * 10000 for crop, info in region_data.items()}  # Convierte Ha a m²
    return land_m2
#Total Area
def total_area(region_data):
    return sum(crop_info['land'] for crop_info in region_data.values())   
# Gross Volume 
def calcular_volumen_bruto(region_data):
    depth_m= calculate_appliedwater(region_data)
    land_m2=area_original_m2(region_data)
    volume_m3 = {}
    for crop in region_data:
        volume = depth_m[crop] * land_m2[crop]
        volume_m3[crop] = round(volume, 2)
    return volume_m3
#Total Gross Volume
def sumar_volumen_total_por_distrito(volume_por_cultivo):
    volumen_total = {}
    for distrito, cultivos in volume_por_cultivo.items():
        total_m3 = sum(cultivos.values())
        total_hm3 = total_m3 / 1e6  # Convertir a hectómetros cúbicos
        volumen_total[distrito] = {'total_m3': round(total_m3, 3), 'total_hm3': round(total_hm3, 4)}
    return volumen_total
#water conveyance efficiency
def calcular_eficiencia_conduccion(region_reportado, volumen_bruto_hm3_dict):
    eficiencia = {}
    for distrito, datos in region_reportado.items():
        volumen_reportado_hm3 = datos["volumen_mm3"] / 1000  # Convertir Mm³ a Hm³
        volumen_bruto_hm3 = volumen_bruto_hm3_dict.get(distrito, {}).get('total_hm3', 3)# Tomar el volumen bruto calculado
        if volumen_bruto_hm3 > 0:  # Evitar división por cero
            eficiencia[distrito] = volumen_bruto_hm3/volumen_reportado_hm3 
    return eficiencia
#Calcular lamina neta
def calcular_lamina_neta(region_data, eficiencia_conduccion_distrito):
    lamina_bruta = calculate_appliedwater(region_data)  # En metros
    lamina_neta = {}
    for cultivo, lamina_m in lamina_bruta.items(): lamina_neta[cultivo] = round(lamina_m / eficiencia_conduccion_distrito, 4)
    return lamina_neta
#Laminas netas totales 
def calcular_lamina_riego_total(land_distritos, laminas_netas):
    lamina_total_por_distrito = {}
    for distrito in land_distritos:
        area_ponderada = sum(land_distritos[distrito][cultivo] * laminas_netas[distrito].get(cultivo, 0)
            for cultivo in land_distritos[distrito])
        total_area = sum(land_distritos[distrito].values())
        if total_area > 0: lamina_total_por_distrito[distrito] = round(area_ponderada / total_area, 4)
    return lamina_total_por_distrito
#Confiabilidad 
def calcular_confiabilidad(lamina_riego_total, region):
    confiabilidad_por_distrito = {}
    for distrito in region:
        lamina_reportada = region[distrito]["lamina_reportada"]
        lamina_modelada = lamina_riego_total.get(distrito, 0)
        if lamina_reportada > 0:
            confiabilidad_por_distrito[distrito] = round(lamina_modelada / lamina_reportada, 4)
    return confiabilidad_por_distrito
#Volumenes 
def calcular_volumen_neto_por_distrito(land_m2, laminas_netas):
    volumen_neto_distrito = {}
    for distrito in land_m2:
        volumen_por_cultivo = {cultivo: land_m2[distrito][cultivo] * laminas_netas[distrito][cultivo] for cultivo in land_m2[distrito]}
        volumen_neto_distrito[distrito] = volumen_por_cultivo
    return volumen_neto_distrito
def calcular_volumen_neto_total(volumen_neto_distrito):
    volumen_total = {}
    for distrito, cultivos in volumen_neto_distrito.items():
        volumen_total[distrito] = round(sum(cultivos.values()), 2)
    return volumen_total
def convertir_volumenes_a_hm3(volumen_total_m3):
    return {distrito: round(v / 1e6, 4) for distrito, v in volumen_total_m3.items()}
#Base Net Returns
def calcular_base_net_returns(data, revenues_distritos):
    base_net_returns = {}
    for distrito in data:
        base_net_returns[distrito] = {}
        for cultivo in data[distrito]:
            costo = data[distrito][cultivo]['cost']
            land_ha = data[distrito][cultivo]['land']
            ingreso = revenues_distritos[distrito][cultivo]
            base_net_returns[distrito][cultivo] = round(land_ha * (ingreso - costo), 2)
    return base_net_returns
#Diccionario de Calculos 
revenues_distrito = {j: calculate_revenues(data[j]) for j in data}
applied_water_m = {j: calculate_appliedwater(data[j]) for j in data}
land_Distritos={j: area_original_m2(data[j]) for j in data}
Total_area_Distritos={j: total_area(data[j]) for j in data}
volume_m3 = {j: calcular_volumen_bruto(data[j]) for j in data}
volumen_total_distritos = sumar_volumen_total_por_distrito(volume_m3)
eficiencia_conduccion = calcular_eficiencia_conduccion(region, volumen_total_distritos)
lamina_neta = {distrito: calcular_lamina_neta(data[distrito], eficiencia_conduccion[distrito]) for distrito in data}
lamina_total_distrito = calcular_lamina_riego_total(land_Distritos, lamina_neta)
confiabilidad_distrito = calcular_confiabilidad(lamina_total_distrito, region)
volumen_neto_distrito = calcular_volumen_neto_por_distrito(land_Distritos, lamina_neta)
volumen_neto_total_m3 = calcular_volumen_neto_total(volumen_neto_distrito)
volumen_neto_total_hm3 = convertir_volumenes_a_hm3(volumen_neto_total_m3)
net_returns = calcular_base_net_returns(data, revenues_distrito)
total_base_net_returns = {distrito: sum(cultivos.values()) for distrito, cultivos in net_returns.items()}
#Step 1 Lineal Optimization 
distritos = ["BConchos"]
models = {}
#Modelo para cada distrito 
for distrito in distritos:
    models[distrito] = ConcreteModel()
    model = models[distrito]
# Definir conjunto de cultivos en el distrito
    model.cultivos = Set(initialize=data[distrito].keys())
# Variables de decisión (área de cultivo en hectáreas)
    model.cultivos_calibracion = Var(model.cultivos, within=NonNegativeReals)
# Parámetros
    model.tierra_ha = Param(model.cultivos, initialize={c: data[distrito][c]['land'] for c in model.cultivos})
    model.total_area = Param(initialize=Total_area_Distritos[distrito])
    model.costos_produccion = Param(model.cultivos, initialize={c: data[distrito][c]['cost'] for c in model.cultivos})
    model.ganancias_produccion = Param(model.cultivos, initialize={c: revenues_distrito[distrito][c] for c in model.cultivos})
    model.laminas_ajustadas = Param(model.cultivos, initialize={c: lamina_neta[distrito][c] for c in model.cultivos})
    model.volumen_distribuido_hm3 = Param(initialize=volumen_hm3_distritos[distrito])
# Función objetivo: Maximizar ganancias netas
    def objetivo_rule(model):
        return sum(model.cultivos_calibracion[c] * (model.ganancias_produccion[c] - model.costos_produccion[c]) for c in model.cultivos)
    model.ganancias_max = Objective(rule=objetivo_rule, sense=maximize)
# Restricción de tierra total
    def land_constraint_total(model):
        return sum(model.cultivos_calibracion[c] for c in model.cultivos) <= model.total_area
    model.restriccion_tierra_total = Constraint(rule=land_constraint_total)
    #model.restriccion_tierra_total = Constraint(expr=land_constraint_total)
# Restricción de tierra por cultivo
    def land_constraint_individual(model, c):
        return model.cultivos_calibracion[c] <= 1.001 * model.tierra_ha[c]
    model.restriccion_tierra_individual = Constraint(model.cultivos, rule=land_constraint_individual)
# Restricción de agua
    def water_constraint_total(model):
        return sum(model.cultivos_calibracion[c] * 10000 * model.laminas_ajustadas[c] for c in model.cultivos) <= model.volumen_distribuido_hm3 * 1e6
    model.restriccion_agua_total = Constraint(rule=water_constraint_total)
# Sufijos para los multiplicadores de Lagrange (valores duales)
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
# Resolver los modelos para cada distrito
opt = SolverFactory('glpk')
resultados = {}
for distrito in distritos:
    resultados[distrito] = opt.solve(models[distrito], tee=True)
#lagrange multiplayer
multiplicadores_tierra_individual = {}
for distrito in distritos:
    model = models[distrito]
    multiplicadores_tierra_individual[distrito] = {}
    for c in model.cultivos:
        multiplicadores_tierra_individual[distrito][c] = model.dual[model.restriccion_tierra_individual[c]]
#Ganancias ajustadas
ganancias_optimizadas = {}
for distrito in distritos:
    model = models[distrito]
    ganancias_optimizadas[distrito] = sum(model.cultivos_calibracion[c].value * (model.ganancias_produccion[c] - model.costos_produccion[c]) 
        for c in model.cultivos)
#Calculo de Coeficientes Alpha y Gamma Step 2 
alpha_por_cultivo = {}
gamma_por_cultivo = {}
for distrito in distritos:
    model = models[distrito]
    alpha_por_cultivo[distrito] = {}
    gamma_por_cultivo[distrito] = {}
    for cultivo in model.cultivos:
        multiplicador_lagrange = model.dual[model.restriccion_tierra_individual[cultivo]]
        alpha = model.costos_produccion[cultivo] - multiplicador_lagrange
        gamma = (2 * multiplicador_lagrange) / model.tierra_ha[cultivo]
        alpha_por_cultivo[distrito][cultivo] = alpha
        gamma_por_cultivo[distrito][cultivo] = gamma
#%%No Lineal Optimization
#%% NO LINEAR OPTIMIZATION
distritos = ["BConchos"]
restricciones_por_distrito = {
    "BConchos": {"nut_area_min": 772*0.85, "fodder_production_min": 20000}}
#Fodder crops 
fodder_crops_por_distrito = {
    "BConchos": ["Avena Forrajera", "Rye Grass", "Sorgo", "Alfalfa"]} 
#Escenarios de disponibilidad 
scenarios_disponibilidad = {distrito: np.arange(0.30, 1.01, 0.05) for distrito in distritos}
#Rendimientos Ajustados
def calcular_rendimientos_por_disponibilidad(data, scenarios_disponibilidad):
    rendimientos_ajustados = {}
    for distrito, cultivos in data.items():
        rendimientos_ajustados[distrito] = {}
        for cultivo, valores in cultivos.items():
            ETm = valores['Evapotranspiration']
            Ky = valores['Ky']
            Ymax = valores['yield']
            rendimientos_ajustados[distrito][cultivo] = {}
            for proporcion in scenarios_disponibilidad[distrito]:
                ETa = ETm * proporcion  # ajuste por disponibilidad
                Y_adj = Ymax * (1 - Ky * (1 - ETa / ETm))
                rendimientos_ajustados[distrito][cultivo][round(proporcion, 2)] = round(Y_adj, 4)
    return rendimientos_ajustados
rendimientos_ajustados = calcular_rendimientos_por_disponibilidad(data, scenarios_disponibilidad)
# Crear listas para almacenar resultados
resultados_areas_cultivo = {}
resultados_beneficios_netos = {}
precios_sombra_agua = {}
modelnolinear = {}
for distrito in distritos:
    for proporcion in scenarios_disponibilidad[distrito]:
       print(f"Escenario con {proporcion*100}% de disponibilidad de agua")
# Crear un modelo de optimización no lineal
       modelnolinear[distrito, proporcion] = ConcreteModel()
       step3 = modelnolinear[distrito, proporcion]
# Conjunto de cultivos a considerar
       step3.cultivos_initialmixt = Set(initialize=list(data[distrito].keys()))
# Variables de decisión
       step3.area_decision = Var(step3.cultivos_initialmixt, within=NonNegativeReals)
# Parámetros
       step3.fodder_crops = Set(initialize=fodder_crops_por_distrito[distrito])
       step3.minimum_fodder_production = Param(initialize=restricciones_por_distrito[distrito]["fodder_production_min"] or 0, within=NonNegativeReals)
       step3.total_area = Param(initialize=Total_area_Distritos[distrito])
       step3.precio_venta= Param(step3.cultivos_initialmixt, initialize={c: data[distrito][c]['price'] for c in data[distrito]})
       step3.laminas_ajustadas = Param(step3.cultivos_initialmixt, initialize={c: lamina_neta[distrito][c] for c in data[distrito]})
       rendimiento_actual = {c: rendimientos_ajustados[distrito][c][round(proporcion, 2)]for c in step3.cultivos_initialmixt}
       step3.yield_production = Param(step3.cultivos_initialmixt, initialize=rendimiento_actual)
       volumen_m3_distritos = {distrito: region[distrito]["volumen_mm3"] * 1e3 for distrito in region}
       step3.volumen_disponible = Param(initialize=volumen_m3_distritos[distrito] * proporcion)
       step3.alpha = Param(step3.cultivos_initialmixt, initialize={c: alpha_por_cultivo[distrito][c] for c in step3.cultivos_initialmixt})
       step3.gamma = Param(step3.cultivos_initialmixt, initialize={c: gamma_por_cultivo[distrito][c] for c in step3.cultivos_initialmixt})
# Establecer la funcion objetivo de maximización de ganancias 
       def objective_rule(step3):
            return sum((step3.precio_venta[crop] * step3.yield_production[crop] * step3.area_decision[crop])
             - (step3.alpha[crop] * step3.area_decision[crop]) - (0.5 * step3.gamma[crop] * step3.area_decision[crop]**2)
            for crop in step3.cultivos_initialmixt)
       step3.ganancias = Objective(rule=objective_rule, sense=maximize)
#Restriccion de tierra
       def restriccion_tierra_total_rule(step3):
            return sum(step3.area_decision[crop] for crop in step3.cultivos_initialmixt) <= step3.total_area
       step3.restriccion_tierra_total = Constraint(rule=restriccion_tierra_total_rule)
#restriccion de los volumenes con escenarios de disponibilidad
       def restriccion_volumen_neto_total_rule(step3):
           volumen_usado = sum(step3.area_decision[c] * 10000 * step3.laminas_ajustadas[c] for c in step3.cultivos_initialmixt)
           return volumen_usado <= step3.volumen_disponible
       step3.restriccion_volumen_neto_total = Constraint(rule=restriccion_volumen_neto_total_rule, doc="Restricción de volumen total de agua")
#restriccion de produccion de forraje
       margen_relajación=0.20
       def restriccion_produccion_fodder_rule(step3):
           produccion_total = sum(step3.area_decision[crop] * step3.yield_production[crop] for crop in step3.fodder_crops)
           meta_total = step3.minimum_fodder_production
           return produccion_total >= meta_total * (1 - margen_relajación)
       step3.restriccion_produccion_fodder = Constraint(rule=restriccion_produccion_fodder_rule)
#Nut 
       def restriccion_area_nut_rule(step3):
           nut_area_min = restricciones_por_distrito[distrito].get("nut_area_min", None)
           if nut_area_min is None: return Constraint.Skip
           return step3.area_decision['NuezdeNogal'] >= nut_area_min
       step3.restriccion_area_nut = Constraint(rule=restriccion_area_nut_rule)
#Lagrange multiplayer 
       step3.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
#Solution 
       solver = SolverFactory('ipopt')
       resultados = solver.solve(step3, tee=True)
#Guardar Resultados 
       clave = (distrito, float(round(proporcion, 2)))
       if (resultados.solver.status == SolverStatus.ok) and (resultados.solver.termination_condition == TerminationCondition.optimal):
          print(f"✅ Solución óptima para {distrito} con disponibilidad {round(proporcion,2)*100}%")
          areas_optimas = {crop: value(step3.area_decision[crop]) for crop in step3.cultivos_initialmixt}
          beneficio_neto = value(step3.ganancias)
          try:
             precio_sombra = step3.dual.get(step3.restriccion_volumen_neto_total, 0)
          except KeyError:
             precio_sombra = None
#Resultados
          resultados_areas_cultivo[clave] = areas_optimas
          resultados_beneficios_netos[clave] = beneficio_neto
          precios_sombra_agua[clave] = precio_sombra

       elif resultados.solver.termination_condition == TerminationCondition.infeasible:
          print(f"⚠️ Modelo infactible para {distrito} con disponibilidad {round(proporcion,2)*100}%")
          resultados_areas_cultivo[clave] = None
          resultados_beneficios_netos[clave] = None
          precios_sombra_agua[clave] = None

       else:
          print(f"❌ Error en la resolución para {distrito} con disponibilidad {round(proporcion,2)*100}%")
          resultados_areas_cultivo[clave] = None
          resultados_beneficios_netos[clave] = None
          precios_sombra_agua[clave] = None          
#%%Print Results 
print("Ingresos por cultivo en Diferentes Distritos:")
for distrito, cultivos in revenues_distrito.items():
    print(f"\nDistrito: {distrito}")
    for cultivo, ingreso in cultivos.items():
        print(f"  {cultivo}: ${ingreso:,.2f}")
print("Lamina de Agua Aplicada A traves de la Evapotranspiranción y el metodo de irrigación:")
for distrito, cultivos in applied_water_m.items():
    print(f"\nDistrito: {distrito}")
    for cultivo, Evapotranspiration in cultivos.items():
        print(f"  {cultivo}: {Evapotranspiration:,.2f}cm")
print("Tierra en m2 en Diferentes Distritos:")
for distrito, cultivos in land_Distritos.items():
    print(f"\nDistrito: {distrito}")
    for cultivo, area in cultivos.items():
        print(f"  {cultivo}: {area:,.0f} m²")
for district, area in Total_area_Distritos.items():
    print(f"Total area {district}: {area} ha")
print("Volumen Bruto de Agua Aplicada por cultivo (m³):")
for distrito, cultivos in volume_m3.items():
    print(f"\nDistrito: {distrito}")
    for cultivo, volumen in cultivos.items():
        print(f"  {cultivo}: {volumen:,.2f} m³")
print("\nVolumen total bruto de agua aplicada por distrito:")
for distrito, valores in volumen_total_distritos.items():
    print(f"\nDistrito: {distrito}")
    print(f"  Total volumen (m³): {valores['total_m3']:,.2f} m³")
    print(f"  Total volumen (Hm³): {valores['total_hm3']:,.4f} Hm³")
print("\nEficiencia de conducción por distrito:")
for distrito, eficiencia in eficiencia_conduccion.items():
    porcentaje = eficiencia * 100
    print(f"  {distrito}: {porcentaje:.2f}%")
print("\nLámina neta aplicada por cultivo (m):")
for distrito, cultivos in lamina_neta.items():
    print(f"\nDistrito: {distrito}")
    for cultivo, lamina in cultivos.items():
        print(f"  {cultivo}: {lamina:.4f} m")
print("\n📏 Lámina total ponderada aplicada por distrito (m):")
for distrito, lamina_total in lamina_total_distrito.items():
    print(f"  {distrito}: {lamina_total:.4f} m")
print("\n✅ Confiabilidad del suministro hídrico por distrito:")
for distrito, confiabilidad in confiabilidad_distrito.items():
    print(f"  {distrito}: {confiabilidad:.2%}")  # Muestra como porcentaje
print("\n💧 Volumen neto aplicado por cultivo (m³):")
for distrito, cultivos in volumen_neto_distrito.items():
    print(f"\nDistrito: {distrito}")
    for cultivo, volumen in cultivos.items():
        print(f"  {cultivo}: {volumen:,.2f} m³")
print("\n💧 Volumen total neto por distrito:")
for distrito in volumen_neto_total_m3:
    print(f"  {distrito}: {volumen_neto_total_m3[distrito]:,.2f} m³ ({volumen_neto_total_hm3[distrito]:.4f} Hm³)")
print("\n💵 Ganancia neta base por cultivo (Net Returns):")
for distrito, cultivos in net_returns.items():
    print(f"\nDistrito: {distrito}")
    for cultivo, ganancia in cultivos.items():
        print(f"  {cultivo}: ${ganancia:,.2f}")
print("\n💵 Ganancia neta total por distrito (en pesos $):")
for distrito, ganancia in total_base_net_returns.items():
    print(f"  {distrito}: ${ganancia:,.2f}")
for distrito in distritos:
    print(f"\nResultados para {distrito}:")
    for c in models[distrito].cultivos:
        print(f"  {c}: {models[distrito].cultivos_calibracion[c].value:.2f} ha")
for distrito, ganancia in ganancias_optimizadas.items():
    print(f"Ganancia optimizada para {distrito}: {ganancia:.2f}")
for distrito, valores in multiplicadores_tierra_individual.items():
    print(f"\nMultiplicadores de Lagrange para {distrito} - Restricción de Tierra Individual:")
    for cultivo, valor in valores.items():
        print(f"  Cultivo {cultivo}: {valor:.6f}")
for distrito in distritos:
    print(f"\nResultados para {distrito}:")
    for cultivo in models[distrito].cultivos:
        print(f"Cultivo: {cultivo} | Alpha: {alpha_por_cultivo[distrito][cultivo]:.4f} | Gamma: {gamma_por_cultivo[distrito][cultivo]:.4f}")
for distrito, cultivos in rendimientos_ajustados.items():
    print(f"\n=== Distrito: {distrito} ===")
    for cultivo, escenarios in cultivos.items():
        print(f"\n  Cultivo: {cultivo}")
        print("    Disponibilidad (%)   |   Rendimiento Ajustado")
        print("    -------------------- | ------------------------")
        for disponibilidad, rendimiento in escenarios.items():
            print(f"        {disponibilidad:.2f}              |     {rendimiento:.2f}")
print("\n\n📊 RESULTADOS FINALES POR ESCENARIO\n" + "="*40)

for clave in resultados_areas_cultivo:
    distrito, proporcion = clave
    print(f"\n🔷 Distrito: {distrito} | Disponibilidad: {proporcion*100:.0f}%")
    
    if resultados_areas_cultivo[clave] is not None:
        print("🌱 Áreas óptimas por cultivo (ha):")
        for cultivo, area in resultados_areas_cultivo[clave].items():
            print(f"   - {cultivo}: {area:.2f}")

        print(f"💰 Beneficio neto: ${resultados_beneficios_netos[clave]:,.2f}")

        precio_sombra = precios_sombra_agua[clave]
        if precio_sombra is not None:
            print(f"💧 Precio sombra del agua: ${precio_sombra:,.4f} por m³")
        else:
            print("💧 Precio sombra del agua: No disponible (no fue vinculante)")
    else:
        print("❌ No se obtuvo una solución óptima para este escenario.")

