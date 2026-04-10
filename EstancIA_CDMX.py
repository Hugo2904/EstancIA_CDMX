import streamlit as st

from model_utils import (
    load_artifacts,
    build_base_input,
    predict_price,
    predict_occupancy,
    calculate_monthly_income,
    market_comparison
)

st.set_page_config(
    page_title="EstancIA CDMX",
    page_icon="🏙️",
    layout="wide"
)

col_logo, col_title = st.columns([1, 7])

with col_logo:
    st.image("Images/logo.png", width=180)

with col_title:
    st.markdown(
        """
        <h1 style='margin-bottom:0;'>
            <span style='color:darkblue;'>Estanc</span><span style='color:green;'>IA</span>
            <span style='color:darkblue;'> CDMX</span>
        </h1>
        <p style='margin-top:0; color: gray; font-size:20px;'>
        📈 Calcula el mejor precio para tu Airbnb y conoce cómo se sitúa frente a alojamientos similares en tu zona.
        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

@st.cache_resource
def get_artifacts():
    return load_artifacts()

artifacts = get_artifacts()

property_type_map = {
    "🏢 Departamento o casa completa": "Departamento o casa completa",
    "🛏️ Habitación privada": "Habitación privada",
    "🏨 Habitación de hotel": "Habitación de hotel",
    "👥 Habitación compartida": "Habitación compartida",
    "📦 Otro": "Otro"
}

zone_options = [
    'Cuajimalpa de Morelos', 'Cuauhtémoc', 'Coyoacán', 'Miguel Hidalgo',
    'Benito Juárez', 'Iztacalco', 'Tlalpan', 'Venustiano Carranza',
    'Gustavo A. Madero', 'Xochimilco', 'Álvaro Obregón', 'Iztapalapa',
    'La Magdalena Contreras', 'Azcapotzalco', 'Tláhuac', 'Milpa Alta'
]

bin_map = {"Sí": 1, "No": 0}

st.subheader("🏠 Características del inmueble")
c1, c2, c3 = st.columns(3)

with c1:
    property_type_label = st.selectbox(
    "¿Qué tipo de alojamiento es el inmueble?",
    list(property_type_map.keys())
)
    property_type = property_type_map[property_type_label]
    
    accommodates = st.slider("Capacidad de huéspedes", 1, 16, 1)

with c2:
    neighbourhood_cleansed = st.selectbox("Alcaldía en la que se ubica el inmueble", zone_options)
    bathrooms = st.slider("Baños", 1, 6, 1)

with c3:
    bedrooms = st.slider("Número de recámaras", 1, 8, 1)
    amenities = st.slider("Número de amenidades", 1, 40, 1)

st.info("""
🧩 ¿Qué son las amenidades?

Las amenidades son atributos adicionales del inmueble que mejoran la experiencia del huésped,
como WiFi, cocina equipada, estacionamiento, aire acondicionado, lavadora o televisión.

Un mayor número de amenidades puede hacer que tu propiedad sea más atractiva y competitiva frente a otras similares.
""")

st.subheader("👤 Tu información como anfitrión en la plataforma de renta")
c4, c5 = st.columns(2)

with c4:
    host_has_profile_pic = bin_map[
        st.selectbox("¿Cuenta con foto de perfil en la plataforma?", list(bin_map.keys()))
    ]
    host_identity_verified = bin_map[
        st.selectbox("¿Ha verificado su identidad en la plataforma?", list(bin_map.keys()))
    ]
    instant_bookable = bin_map[
        st.selectbox("¿El inmueble cuenta con reserva inmediata?", list(bin_map.keys()))
    ]

with c5:
    host_verifications = st.number_input(
        "¿Mediante cuántos medios de comunicación se puede poner en contacto con usted?",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    host_age = st.number_input(
        "¿Cuántos años ha puesto en renta el inmueble?",
        min_value=0,
        max_value=17,
        value=1,
        step=1
    )

    reviews = st.number_input(
        "¿Cuántos reviews ha recibido el inmueble?",
        min_value=0,
        max_value=1434,
        value=30,
        step=1
    )

st.subheader("📅 Reglas del alojamiento")
c6, c7 = st.columns(2)

with c6:
    min_nights = st.number_input(
        "Mínimo de noches para reservar",
        min_value=1,
        max_value=365,
        value=1,
        step=1
    )

    max_nights = st.number_input(
        "Máximo de noches para reservar",
        min_value=1,
        max_value=365,
        value=1,
        step=1
    )

with c7:
    availability_per_month = st.number_input(
        "En un periodo de 30 días, ¿cuántos días está disponible el inmueble para renta del público?",
        min_value=0,
        max_value=30,
        value=30,
        step=1
    )

st.subheader("💰 Precio inicial")
precio_usuario = st.number_input(
    "¿En cuánto piensa ofertar el inmueble por noche?",
    min_value=100.0,
    max_value=50000.0,
    value=1500.0,
    step=50.0
)

df_base = build_base_input(
    property_type=property_type,
    neighbourhood_cleansed=neighbourhood_cleansed,
    host_age=host_age,
    host_verifications=host_verifications,
    host_has_profile_pic=host_has_profile_pic,
    host_identity_verified=host_identity_verified,
    accommodates=accommodates,
    bathrooms=bathrooms,
    bedrooms=bedrooms,
    amenities=amenities,
    minimum_nights=min_nights,
    maximum_nights=max_nights,
    number_of_reviews=reviews,
    instant_bookable=instant_bookable,
    availability_30=availability_per_month
)

try:
    precio_sugerido = predict_price(df_base, artifacts)
except Exception as e:
    st.error(f"Error al predecir el precio: {e}")
    st.stop()

precio_base_slider = max(100, int(round(precio_usuario / 50) * 50))
precio_min_slider = max(100, int(precio_sugerido * 0.2))
precio_max_slider = max(precio_min_slider + 50, int(precio_sugerido * 2))

if precio_base_slider < precio_min_slider:
    precio_base_slider = precio_min_slider
if precio_base_slider > precio_max_slider:
    precio_base_slider = precio_max_slider

st.subheader("🎛️ Sensibilidad de precio")
precio_activo = st.slider(
    "Ajuste el precio por noche",
    min_value=precio_min_slider,
    max_value=precio_max_slider,
    value=precio_base_slider,
    step=50
)

try:
    ocupacion_estimada = predict_occupancy(df_base, precio_activo, artifacts)
    ingreso_estimado = calculate_monthly_income(precio_activo, ocupacion_estimada,days=availability_per_month)

    comparacion = market_comparison(
        df_base=df_base,
        user_price=precio_activo,
        artifacts=artifacts
    )
except Exception as e:
    st.error(f"Error al calcular métricas adicionales: {e}")
    st.stop()

st.subheader("📊 Resultados")
r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric("Precio sugerido", f"${precio_sugerido:,.0f} MXN")

with r2:
    delta_abs = precio_activo - precio_sugerido
    st.metric("Precio evaluado", f"${precio_activo:,.0f} MXN", delta=f"${delta_abs:,.0f}")

with r3:
    st.metric("Ocupación estimada", f"{ocupacion_estimada:.1%}")

with r4:
    st.metric("Ingreso mensual estimado", f"${ingreso_estimado:,.0f} MXN")


st.subheader("🏪 Comparación del precio evaluado con el mercado")

st.info("""
🔎 ¿Cómo hacemos esta comparación?

Tomamos como referencia alojamientos de la misma zona y del mismo tipo de propiedad.

Dentro de ese grupo, damos más peso a los alojamientos que más se parecen al tuyo
en capacidad, número de baños y número de recámaras, para que la comparación sea
más útil y representativa.
""")

if comparacion.get("missing_cols"):
    st.error(f"No se pudo realizar la comparación porque faltan columnas en market_data.pkl: {comparacion['missing_cols']}")

elif comparacion["n_comparables"] > 0:

    st.caption(
        f"Comparando con alojamientos en {neighbourhood_cleansed}, "
        f"del tipo {property_type}, priorizando propiedades parecidas en capacidad, baños y recámaras."
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Número de alojamientos similares", comparacion["n_comparables"])

    with m2:
        st.metric("Precio promedio del mercado", f"${comparacion['market_avg_price']:,.0f} MXN")

    with m3:
        st.metric("Diferencia vs mercado", f"{comparacion['price_diff_pct']:.1f}%")

    with m4:
        st.metric("Percentil de precio", f"{comparacion['price_percentile']:.0f}%")

    st.caption("El percentil indica qué porcentaje de alojamientos comparables tiene un precio menor al tuyo.")
    st.caption("""La diferencia vs mercado indica cómo se compara tu precio evaluado con el de otros alojamientos similares en tu zona.
                  Un valor positivo significa que estás cobrando más que el promedio; uno negativo, que estás cobrando menos.""")

    if comparacion["position"] == "Por debajo del mercado":
        st.success(f"Posicionamiento: {comparacion['position']}")
    elif comparacion["position"] == "Por encima del mercado":
        st.warning(f"Posicionamiento: {comparacion['position']}")
    else:
        st.info(f"Posicionamiento: {comparacion['position']}")

    st.subheader("📊 Perfil del inmueble vs mercado")
    st.caption("Comparación de su inmueble contra propiedades similares en el mercado.")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Amenidades",
            f"{comparacion['user_amenities']:.0f}",
            delta=f"{comparacion['amenities_diff']:+.0f}"
        )
        st.caption(f"Promedio mercado: {comparacion['avg_amenities']:.0f}")

    with c2:
        st.metric(
            "Antigüedad",
            f"{comparacion['user_host_age']:.1f} años",
            delta=f"{comparacion['host_age_diff']:+.1f}"
        )
        st.caption(f"Promedio mercado: {comparacion['avg_host_age']:.1f}")

    with c3:
        st.metric(
            "Baños",
            f"{comparacion['user_bathrooms']:.1f}",
            delta=f"{comparacion['bathrooms_diff']:+.1f}"
        )
        st.caption(f"Promedio mercado: {comparacion['avg_bathrooms']:.1f}")

    with c4:
        st.metric(
            "Recámaras",
            f"{comparacion['user_bedrooms']:.0f}",
            delta=f"{comparacion['bedrooms_diff']:+.0f}"
        )
        st.caption(f"Promedio mercado: {comparacion['avg_bedrooms']:.0f}")

    with c5:
        st.metric(
            "Capacidad de huéspedes",
            f"{comparacion['user_accommodates']:.0f}",
            delta=f"{comparacion['accommodates_diff']:+.0f}"
        )
        st.caption(f"Promedio mercado: {comparacion['avg_accommodates']:.0f}")

    st.subheader("🧠 Datos relevantes acerca de su inmueble")
    st.caption("Estos hallazgos se construyen comparando su propiedad contra alojamientos similares en la zona.")

    for insight in comparacion["insights"]:
        st.write(f"• {insight}")

else:
    st.info("No se encontraron alojamientos comparables para esta configuración.")
