"""
Identidad visual del chat: paleta slate + azul corporativo.

Diseño minimalista con contrastes WCAG AA:
    - Texto principal #0F172A sobre fondo #F8FAFC → contraste 17:1.
    - Acento azul #1E40AF para elementos interactivos.
    - Burbujas con jerarquía clara (usuario tinta sólida, bot blanco con borde).

Las constantes de la paleta están separadas del tema para que se pueda
afinar la marca sin bucear en el CSS. `build_theme()` devuelve un
`gr.themes.Base`; `CUSTOM_CSS` inyecta tipografías, layout y burbujas.
"""

import gradio as gr


# ---------------------------------------------------------------------------
# Paleta — slate + azul corporativo
# ---------------------------------------------------------------------------
# Texto
TEXT = "#0F172A"          # slate-900 — texto principal
TEXT_SOFT = "#475569"     # slate-600 — texto secundario / subtítulos
TEXT_MUTED = "#64748B"    # slate-500 — placeholders, etiquetas

# Superficies
BG = "#F8FAFC"            # slate-50 — fondo de la página
SURFACE = "#FFFFFF"       # blanco — cajas, chat, inputs
SURFACE_ALT = "#F1F5F9"   # slate-100 — hovers sutiles

# Bordes
BORDER = "#E2E8F0"        # slate-200 — bordes por defecto
BORDER_STRONG = "#CBD5E1" # slate-300 — bordes en hover/focus base

# Acento
ACCENT = "#1E40AF"        # blue-800 — botones, enlaces, elementos activos
ACCENT_HOVER = "#1D4ED8"  # blue-700
ACCENT_SOFT = "#EFF6FF"   # blue-50 — fondos sutiles del acento

# Burbuja del usuario
USER_BG = "#0F172A"       # slate-900
USER_TEXT = "#FFFFFF"


def build_theme() -> gr.themes.Base:
    """Construye el tema Gradio con la paleta slate + azul."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#EFF6FF", c100="#DBEAFE", c200="#BFDBFE", c300="#93C5FD",
            c400="#60A5FA", c500="#3B82F6", c600=ACCENT_HOVER, c700="#1D4ED8",
            c800=ACCENT, c900="#1E3A8A", c950="#172554",
        ),
        neutral_hue=gr.themes.Color(
            c50=BG, c100=SURFACE_ALT, c200=BORDER, c300=BORDER_STRONG,
            c400="#94A3B8", c500=TEXT_MUTED, c600=TEXT_SOFT, c700="#334155",
            c800="#1E293B", c900=TEXT, c950="#020617",
        ),
        font=("Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"),
        font_mono=("JetBrains Mono", "ui-monospace", "monospace"),
    ).set(
        body_background_fill=BG,
        body_text_color=TEXT,
        background_fill_primary=SURFACE,
        background_fill_secondary=BG,
        border_color_primary=BORDER,
        block_background_fill=SURFACE,
        block_border_color=BORDER,
        block_border_width="1px",
        block_radius="12px",
        button_primary_background_fill=ACCENT,
        button_primary_background_fill_hover=ACCENT_HOVER,
        button_primary_text_color=USER_TEXT,
        button_primary_border_color=ACCENT,
        button_secondary_background_fill=SURFACE,
        button_secondary_background_fill_hover=SURFACE_ALT,
        button_secondary_text_color=TEXT,
        button_secondary_border_color=BORDER,
        input_background_fill=SURFACE,
        input_border_color=BORDER,
        input_border_color_focus=ACCENT,
        input_radius="10px",
    )


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Forzamos modo claro: si el navegador tiene prefers-color-scheme: dark,
   Gradio repinta variables internas y rompe el contraste. */
:root, html, body, .gradio-container {
    color-scheme: light !important;
}

/* El body de la página HF (fuera del container Gradio) hereda el color de
   sistema. En el subdominio *.hf.space queda oscuro por defecto y crea
   franjas negras a los lados. Lo igualamos al fondo del container. */
html, body {
    background-color: #F8FAFC !important;
    margin: 0 !important;
    min-height: 100vh !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 1.25rem 1.5rem 1.5rem !important;
    background-color: #F8FAFC !important;
    background-image: none !important;
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    color: #0F172A !important;
}

/* Cabecera del ChatInterface */
.gradio-container h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: clamp(1.75rem, 3vw, 2.25rem) !important;
    line-height: 1.15 !important;
    letter-spacing: -0.02em !important;
    color: #0F172A !important;
    margin: 0 0 0.4rem !important;
    position: relative;
}

.gradio-container h1::before {
    content: "PORTFOLIO · RAG CONVERSACIONAL";
    display: block;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #1E40AF;
    margin-bottom: 0.5rem;
}

/* Subtítulo (description del ChatInterface). Por defecto, todos los `.prose`
   reciben el color del subtítulo. La regla del chat (más abajo) tiene mayor
   especificidad para sobrescribirlo dentro de las burbujas. */
.gradio-container .prose,
.gradio-container .prose p {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.55 !important;
    color: #475569 !important;
    max-width: 70ch;
    margin: 0 0 1rem !important;
    font-style: normal !important;
    opacity: 1 !important;
}

/* Selección de texto: usa la paleta azul corporativa en lugar del morado
   por defecto del navegador. */
::selection {
    background-color: rgba(59, 130, 246, 0.35) !important;
    color: inherit !important;
}
::-moz-selection {
    background-color: rgba(59, 130, 246, 0.35) !important;
    color: inherit !important;
}

/* Reset duro del `.prose` dentro del chat: el markdown del bot va en
   prose por defecto y heredaba estilos de subtítulo. */
.chatbot .prose,
.chatbot .prose p,
[data-testid="chatbot"] .prose,
[data-testid="chatbot"] .prose p,
.message-content .prose,
.message-content .prose p,
.bubble-wrap .prose,
.bubble-wrap .prose p,
.message .prose,
.message .prose p {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-style: normal !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    color: #0F172A !important;
    max-width: none !important;
    margin: 0 !important;
}

/* Contenedor del chat */
.chatbot,
[data-testid="chatbot"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02) !important;
}

/* Avatares */
.avatar-container,
[data-testid="avatar"],
.message .avatar,
.message-row .avatar {
    width: 34px !important;
    height: 34px !important;
    flex-shrink: 0 !important;
}

.avatar-container img,
[data-testid="avatar"] img,
.message .avatar img,
.message-row .avatar img {
    border-radius: 50% !important;
    border: 1px solid #E2E8F0 !important;
    width: 34px !important;
    height: 34px !important;
    object-fit: cover !important;
}

/* Burbujas — base */
.message-wrap, .message, .bubble, .bubble-wrap {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* Burbuja del bot: blanca con borde slate */
.bot .message-bubble-border,
.bot-row .message-bubble-border,
.message.bot,
[data-testid="bot"],
[data-testid="bot"] .message-bubble-border,
[data-testid="bot-message"],
[role="log"] [data-bot="true"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 4px 12px 12px 12px !important;
    color: #0F172A !important;
}

.bot .message-bubble-border *,
[data-testid="bot"] *,
[data-testid="bot-message"] * {
    color: #0F172A !important;
    background: transparent !important;
}

/* Burbuja del usuario: tinta sólida con texto blanco.
   Selectores con `[class*="user"]` para capturar las clases hash de Svelte
   que Gradio 6 genera (.user-row.svelte-1a2b3c4, etc.) y que tienen mayor
   especificidad que selectores con clase simple. */
.user .message-bubble-border,
.user-row .message-bubble-border,
.message.user,
[data-testid="user"],
[data-testid="user"] .message-bubble-border,
[data-testid="user-message"],
.gradio-container [class*="user-row"],
.gradio-container [class*="user-row"] [class*="message"] {
    background: #0F172A !important;
    background-color: #0F172A !important;
    border: 1px solid #0F172A !important;
    border-radius: 12px 4px 12px 12px !important;
    color: #FFFFFF !important;
}

/* Texto blanco forzado en TODOS los descendientes de la burbuja del usuario.
   El selector `[class*="user"]` matchea cualquier clase que contenga "user"
   (incluidas .user-row.svelte-XXX), evitando la guerra de especificidad
   contra el CSS interno de Gradio. */
.user .message-bubble-border *,
.user-row *,
[data-testid="user"] *,
[data-testid="user-message"] *,
.gradio-container [class*="user-row"] *,
.gradio-container [class*="user"] [class*="message"] *,
.gradio-container [class*="user"] [class*="prose"],
.gradio-container [class*="user"] [class*="prose"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background: transparent !important;
}

/* Textarea / input — fondo oscuro con texto blanco. Misma estética que la
   burbuja del usuario una vez enviado el mensaje: lo que el reclutador
   escribe queda visualmente igual a lo que ya envió. */
textarea, input[type="text"],
.gradio-container textarea,
.gradio-container input[type="text"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.95rem !important;
    background: #0F172A !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    transition: border-color 150ms ease, box-shadow 150ms ease !important;
}

textarea:focus, input[type="text"]:focus {
    outline: none !important;
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25) !important;
}

textarea::placeholder, input::placeholder {
    color: #94A3B8 !important;
    font-style: normal !important;
    opacity: 1 !important;
}

/* Botones primarios */
button.primary, button[variant="primary"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    background: #1E40AF !important;
    color: #FFFFFF !important;
    border: 1px solid #1E40AF !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    transition: background 150ms ease, border-color 150ms ease !important;
}

button.primary:hover, button[variant="primary"]:hover {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

button.secondary, button[variant="secondary"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    background: #FFFFFF !important;
    color: #475569 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

button.secondary:hover, button[variant="secondary"]:hover {
    background: #F1F5F9 !important;
    color: #0F172A !important;
    border-color: #CBD5E1 !important;
}

/* Ejemplos — píldoras con contraste alto. Selectores múltiples y `*` para
   sobreescribir todos los descendientes con clases internas .svelte-* que
   Gradio renderiza dentro del botón. */
.examples,
[data-testid="examples"],
.gradio-container .examples,
.gradio-container [data-testid="examples"] {
    margin-top: 1rem;
}

.examples button,
[data-testid="examples"] button,
.gradio-container .examples button,
.gradio-container [data-testid="examples"] button,
.examples [role="button"],
[data-testid="example"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 999px !important;
    padding: 9px 18px !important;
    transition: all 150ms ease !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06) !important;
}

/* Forzar el color del texto en TODOS los descendientes del botón
   (Gradio mete spans/divs internos que se quedan con su color por defecto). */
.examples button *,
[data-testid="examples"] button *,
.gradio-container .examples button *,
.examples [role="button"] *,
[data-testid="example"] * {
    color: #1E293B !important;
    -webkit-text-fill-color: #1E293B !important;
    background: transparent !important;
}

.examples button:hover,
[data-testid="examples"] button:hover,
.gradio-container .examples button:hover,
.examples [role="button"]:hover {
    background: #1E40AF !important;
    background-color: #1E40AF !important;
    border-color: #1E40AF !important;
}

.examples button:hover *,
[data-testid="examples"] button:hover *,
.examples [role="button"]:hover * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Markdown del bot: enlaces, separadores, énfasis */
.bot a, [data-testid="bot"] a, [data-testid="bot-message"] a {
    color: #1E40AF !important;
    text-decoration: underline;
    text-decoration-color: rgba(30, 64, 175, 0.4);
    text-underline-offset: 3px;
    transition: text-decoration-color 150ms ease;
}

.bot a:hover, [data-testid="bot"] a:hover {
    text-decoration-color: #1E40AF;
}

.bot hr, [data-testid="bot"] hr, [data-testid="bot-message"] hr {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 1rem 0 0.75rem !important;
}

.bot strong, [data-testid="bot"] strong, [data-testid="bot-message"] strong {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 600 !important;
    color: #0F172A !important;
}

/* Quita el "Built with Gradio" del pie */
footer { display: none !important; }

/* Scrollbars discretas */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* Aparición suave de mensajes */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

.message-wrap > .message,
.bubble {
    animation: fadeIn 200ms ease-out both;
}
"""
