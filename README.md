# Optimization Conchos Basin 

This tutorial primarily aims to guide beginner readers through the practical implementation of Positive Mathematical Programming (PMP), as introduced by Medellín-Azuara, Harou, and Howitt (2010), using the Python programming language and the Pyomo environment. This repository includes the source codes for the four irrigation districts analyzed in this study—Delicias, Bajo Conchos, Florido, and Alto Conchos—as well as for each of their possible coalitions, which are detailed below.

Coalition
{Del}
{BC}
{Flor}
{AC}
{DEL,BC}
{DEL,FLOR}
{DEL,AC}
{BC,FLOR}
{BC,AC}
{FLOR, AC}
{DEL,BC,FLOR}
{DEL,BC,AC}
{DEL,FLOR,AC}
{BC,FLOR,AC}
{DEL, BC, FLOR, AC}

With the coalitions already in place, and in order to generate adequate public policies to address the severe conditions observed in recent decades, it is crucial to establish different scenarios that foster cooperation and new institutional arrangements. These will be supported by a cooperative game theory model.
For the correct elaboration of these scenarios, key factors should be considered, such as:
Price changes
Water availability
Specific crop requirements in each irrigation district.
The need to deliver volumes of water to the international treaty of 1944.
The change in crop yields as a result of the water deficit, based on studies by organizations such as FAO, CILA, Firas and CONAGUA.

The base information inputs needed to start with the proposed code are as follows:
#Data 
region = {"Delicias": {"volumen_mm3": 976309.62, "lamina_reportada": 1.44},
          "BConchos": {"volumen_mm3": 79206.52, "lamina_reportada": 2.46},
          "Florido": {"volumen_mm3": 64095.84, "lamina_reportada": 1.74},
          "Aconchos": {"volumen_mm3": 82425.73, "lamina_reportada": .74}} # en m 
volumen_hm3_distritos = {distrito: region[distrito]["volumen_mm3"] / 1_000 for distrito in region}

data = {
    'Delicias': {
        'Cacahuate': {'price': 11713, 'yield':3 , 'cost': 32170, 'land': 4041, 'Evapotranspiration':54.96,'IRMETHOD':'Asperción', 'METHODEF':.75, 'Ky':.7},#METHODEF es la eficiencia del metodo de irrigación
        'Cebolla': {'price': 5070, 'yield': 85, 'cost': 136797, 'land': 1758, 'Evapotranspiration':102,'IRMETHOD':'GOTEO', 'METHODEF':.90, 'Ky':1.1},
        'Chile': {'price': 5773, 'yield': 50, 'cost': 132680, 'land': 4854, 'Evapotranspiration':43.25,'IRMETHOD':'Tradicional', 'METHODEF':.60, 'Ky':1.1},
        'MaizForrajero': {'price': 3600, 'yield': 75, 'cost': 40070, 'land': 8416, 'Evapotranspiration':65.37,'IRMETHOD':'Tradicional', 'METHODEF':.60, 'Ky':1.25},
        'Sandia': {'price': 2000, 'yield': 56, 'cost': 77314, 'land': 5129, 'Evapotranspiration':37.91,'IRMETHOD':'GOTEO', 'METHODEF':.90, 'Ky':1.1},
        'Alfalfa': {'price': 2266, 'yield': 65, 'cost': 32364, 'land': 32294, 'Evapotranspiration':102.26,'IRMETHOD':'Tradicional', 'METHODEF':.60, 'Ky':1.1},
        'NuezdeNogal': {'price': 72522, 'yield': 2, 'cost': 94148, 'land': 14202, 'Evapotranspiration':119.31,'IRMETHOD':'Asperción', 'METHODEF':.75, 'Ky':1.2}},
    'BConchos': {
        'Avena Forrajera': {'price':6113 , 'yield': 31, 'cost': 23837 , 'land':427 , 'Evapotranspiration':16.2,'IRMETHOD':'Tradicional','METHODEF':.6, 'Ky':.7},
        'Rye Grass': {'price': 906 , 'yield': 75, 'cost': 11733 , 'land':190 , 'Evapotranspiration':38.3,'IRMETHOD':'tradicional','METHODEF':.6, 'Ky':.8},
        'Algodon': {'price': 29680 , 'yield': 5, 'cost':47071 , 'land':106 , 'Evapotranspiration':80.80,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':.85},
        'Sorgo': {'price': 680, 'yield': 78, 'cost': 29616 , 'land': 247, 'Evapotranspiration':54.90,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':.9},
        'Alfalfa': {'price':2266 , 'yield': 84, 'cost': 32364, 'land': 1531 , 'Evapotranspiration':100.17,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':1.1},
        'NuezdeNogal': {'price':72522 , 'yield': 2.2, 'cost': 94148, 'land': 777 , 'Evapotranspiration':116.86,'IRMETHOD':'Asperción','METHODEF':.75, 'Ky':1.2}},
    'Florido': {
        'Avena Forrajera': {'price':6113 , 'yield':44 , 'cost': 23837 , 'land':177 , 'Evapotranspiration':39.75,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':1.10},
        'Chile': {'price': 5773 , 'yield': 42, 'cost': 132680 , 'land':104 , 'Evapotranspiration':40.80,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':1.1},
        'MaizForrajero': {'price': 3600, 'yield': 46, 'cost':40070 , 'land':427 , 'Evapotranspiration':60.95,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':1.25},
        'Sorgo': {'price': 680, 'yield': 44, 'cost': 29616 , 'land': 231, 'Evapotranspiration':61.69,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':.9},
        'Alfalfa': {'price':2266 , 'yield': 46.0, 'cost': 32364, 'land': 1909 , 'Evapotranspiration':95.22,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':1.1},
        'NuezdeNogal': {'price':72522 , 'yield': 2, 'cost': 94148, 'land': 844 , 'Evapotranspiration':111.10,'IRMETHOD':'Asperción','METHODEF':.75, 'Ky':1.2}},
    'Aconchos': {
        'Alfalfa': {'price': 2266, 'yield': 77, 'cost': 32364, 'land': 2920, 'Evapotranspiration':98.7,'IRMETHOD':'Tradicional','METHODEF':.60, 'Ky':1.1},
        'NuezdeNogal': {'price': 72522, 'yield': 2.5, 'cost': 94148, 'land': 8264, 'Evapotranspiration':115.1,'IRMETHOD':'Asperción','METHODEF':.75, 'Ky':1.2}}}