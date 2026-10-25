# Root Finder & Numerical Methods Visualizer
Foundation of Mathematics, Unit 2: Numerical Methods

An interactive web app (built with Streamlit) that finds roots of f(x) = 0 using
Bisection, Regula-Falsi, Newton-Raphson and Secant, then compares their speed and error.

## Run it on your computer (5 steps)
1. Install Python 3.10 or newer from https://www.python.org/downloads/ (tick "Add Python to PATH").
2. Put `app.py`, `requirements.txt` and the `.streamlit` folder in one folder, e.g. `root_finder_app`.
3. Open a terminal in that folder and create a virtual environment (optional but tidy):
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install the libraries: `pip install -r requirements.txt`
5. Start the app: `streamlit run app.py`  (your browser opens at http://localhost:8501)

## Put it online for free (Streamlit Community Cloud) so you can email a link
1. Create a free GitHub account and a new public repository (e.g. `root-finder-visualizer`).
2. Upload `app.py`, `requirements.txt`, `.streamlit/config.toml` and this README to the repository.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click "Create app" > "Deploy a public app from GitHub".
5. Choose your repository, branch `main`, main file path `app.py`, then click Deploy.
6. After a minute you get a link like `https://your-name-root-finder.streamlit.app`. Put it in your email.

## What to say in the email
- Objective: compare four root-finding methods and show their errors and convergence visually.
- Unit and concepts: Unit 2, Numerical Methods (absolute/relative/percentage error, Bisection,
  Regula-Falsi, Newton-Raphson, Secant, order of convergence).
- Computing application: git bisect (binary search), square roots in hardware, optimisation in
  machine learning, ray-surface intersection in graphics.
- What I learned: see the "What I learned" tab in the app (rewrite it in your own words).

## Files
- `app.py` the whole application
- `requirements.txt` libraries to install
- `.streamlit/config.toml` colour theme
