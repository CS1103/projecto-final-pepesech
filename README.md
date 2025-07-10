[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación de una red neuronal multicapa en C++ desde cero, enfocada en resolver un problema de clasificación binaria en un espacio bidimensional. El modelo debe aprender a determinar si un punto (x, y) pertenece o no a una región con forma de anillo.

La región está definida por dos radios: r_min y r_max, clasificando como clase 1 aquellos puntos que cumplen la condición:

r_min² < x² + y² < r_max²

Por ejemplo, con r_min = 0.5 y r_max = 1.0, los puntos válidos se ubican en el rango:

0.25 < x² + y² < 1.0

Entradas: 2 valores (x, y)

Salida esperada: 1 valor (0 = fuera del anillo, 1 = dentro del anillo)

Ventaja: Se puede graficar fácilmente tanto el dataset como las predicciones del modelo.



### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `group_3_custom_name`
* **Integrantes**:

  * Fatima Isabella Pacheco Vera – 202410182 (Responsable de investigación teórica)
  * Daniela Valentina Villacorta Sotelo –  202410253 (Desarrollo de la arquitectura)La 
  * Valentin Tuesta – 202410251 (Implementación del modelo)
  * Emma Anderson Gonzalez  – 202410607 (Pruebas y benchmarking)
  * Alumno E – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido de ejemplo**:

1. Fundamentos de redes neuronales artificiales

2. Clasificación binaria y no lineal

3. Funciones de activación (ReLU, Sigmoid)

4. Algoritmo de retropropagación

5. Uso de redes densas en tareas de separación geométrica

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ``
proyecto-final/
├── data/
│   └── dataset.csv
├── src/
│   ├── include/
│   │   ├── neural_network.h
│   │   ├── nn_activation.h
│   │   ├── nn_dense.h
│   │   ├── nn_interfaces.h
│   │   ├── nn_loss.h
│   │   ├── nn_optimizer.h
│   │   └── tensor.h
│   └── main.cpp
├── tools/
│   └── gen_dataset.cpp
├── .gitignore
├── CMakeLists.txt
└── README.md

  ``

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---
La 
### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 63.5892 segundos
  * Precisión final: 71.65%.

* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.

* **Mejoras futuras**:

  * Implementar visualización integrada en C++

---

### 5. Trabajo en equipo

| Tarea                   | Miembro  | Rol                               |
| ----------------------- | -------- | --------------------------------- |
| Investigación teórica   | Fatima   | Documentación y marco teórico     |
| Diseño de arquitectura  | Daniela  | Organización de archivos y clases |
| Lógica del clasificador | Valentin | Implementación de red y condición |
| Pruebas y benchmarking  | Emma     | Métricas y validación final       |
| Documentación y dataset | Alumno E | README y generación de datos      |



---

### 6. Conclusiones

Se logró construir un clasificador funcional basado en red neuronal que diferencia puntos dentro o fuera de una región en forma de anillo.

El proyecto permitió afianzar conocimientos sobre retropropagación, estructuras modulares y compilación avanzada en C++.

El trabajo colaborativo fue clave para dividir responsabilidades y mantener un flujo ordenado de desarrollo.

---

### 7. Bibliografía

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
