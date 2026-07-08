# Guía de setup: GitHub + VS Code desde cero

Sigue estos pasos **en orden**. No te saltes ninguno la primera vez.

---

## PASO 1 — Instalar Git en tu ordenador

1. Ve a https://git-scm.com/download/win
2. Descarga el instalador y ejecútalo
3. Durante la instalación deja **todas las opciones por defecto** (dale a Next en todo)
4. Cuando termine, abre el **Símbolo del sistema** (busca "cmd" en el buscador de Windows) y escribe:
   ```
   git --version
   ```
   Debería aparecer algo como `git version 2.44.0`. Si aparece, Git está instalado correctamente.

---

## PASO 2 — Crear cuenta en GitHub

1. Ve a https://github.com
2. Haz clic en **Sign up**
3. Usa tu email y crea una contraseña
4. Elige el plan **Free** (es suficiente para todo el TFM)
5. Verifica tu email cuando te llegue el correo de confirmación

---

## PASO 3 — Configurar Git con tu identidad

Abre el **Símbolo del sistema** (cmd) y ejecuta estos dos comandos (cambia el nombre y email por los tuyos):

```
git config --global user.name "Meritxell Abellan"
git config --global user.email "abellan.meritxell@gmail.com"
```

Solo hay que hacerlo una vez.

---

## PASO 4 — Crear el repositorio en GitHub

1. En GitHub, haz clic en el botón verde **New** (o el símbolo **+** arriba a la derecha → New repository)
2. Rellena así:
   - **Repository name:** `TFM-accidentes-trafico`
   - **Description:** `TFM - Predicción de gravedad en accidentes de tráfico en Madrid`
   - **Visibility:** Private (o Public si quieres que sea visible en tu CV)
   - **NO marques** ninguna de las opciones de inicializar (sin README, sin .gitignore, sin licencia) — ya los tienes tú
3. Haz clic en **Create repository**
4. GitHub te mostrará una pantalla con instrucciones — **no cierres esa página**, la necesitas en el Paso 6

---

## PASO 5 — Crear la carpeta del proyecto en tu ordenador

1. Crea una carpeta en tu ordenador donde quieras tener el proyecto, por ejemplo:
   ```
   C:\Users\TuNombre\Documents\TFM-accidentes-trafico
   ```
2. Dentro de esa carpeta, crea estas subcarpetas:
   ```
   data\raw\
   data\processed\
   notebooks\
   src\
   figures\
   models\
   ```
3. Copia en la carpeta raíz los archivos que te he preparado:
   - `README.md`
   - `requirements.txt`
   - `.gitignore`
4. Copia tu dataset Excel en `data\raw\`

---

## PASO 6 — Conectar la carpeta con GitHub

Abre cmd, navega a tu carpeta y ejecuta los comandos uno a uno:

```bash
# Ir a la carpeta del proyecto
cd C:\Users\TuNombre\Documents\TFM-accidentes-trafico

# Inicializar Git
git init

# Añadir todos los archivos
git add .

# Primer commit
git commit -m "Estructura inicial del proyecto"

# Conectar con GitHub (copia la URL de tu repositorio de la pantalla que dejaste abierta)
git remote add origin https://github.com/TU_USUARIO/TFM-accidentes-trafico.git

# Subir
git branch -M main
git push -u origin main
```

Cuando ejecutes `git push`, GitHub te pedirá que inicies sesión. Sigue las instrucciones que aparezcan (abrirá el navegador para autenticarte).

---

## PASO 7 — Instalar VS Code y extensiones

1. Descarga VS Code desde https://code.visualstudio.com
2. Instálalo con opciones por defecto
3. Abre VS Code y en el panel izquierdo (el icono de cuadrados) busca e instala estas extensiones:
   - **Python** (Microsoft)
   - **Jupyter** (Microsoft)
   - **GitLens** (opcional pero muy útil para ver el historial de cambios)

---

## PASO 8 — Crear el entorno virtual de Python

Desde la terminal integrada de VS Code (menú Terminal → New Terminal):

```bash
# Crear entorno virtual
python -m venv venv

# Activarlo (Windows)
venv\Scripts\activate

# Instalar librerías
pip install -r requirements.txt
```

Cuando el entorno esté activo verás `(venv)` al principio de la línea de comandos.

---

## Los 5 comandos Git que usarás cada día

Cada vez que termines de trabajar:

```bash
git add .
git commit -m "Descripción breve de lo que hiciste"
git push
```

Para ver el estado de tus cambios:
```bash
git status
```

Para ver el historial de commits:
```bash
git log --oneline
```

Eso es todo lo que necesitas saber de Git para este TFM.

---

## ¿Cuándo hacer un commit?

No hay que hacerlo cada cinco minutos. Hazlo cuando termines algo concreto:
- "EDA primera exploración terminada"
- "Limpieza de nulos completada"
- "Modelo baseline entrenado"

Piénsalo como guardar una versión a la que podrías volver si algo se rompe.
