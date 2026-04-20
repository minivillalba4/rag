# Documentos fuente

Coloca aquí los ficheros que quieres que el RAG conozca:

- `CV.pdf` — tu CV en PDF.
- `proyectos.md` — descripción de tus proyectos más relevantes.
- Cualquier otro `.pdf`, `.md` o `.txt` se indexa automáticamente.

Tras añadir o modificar archivos, regenera el índice:

```bash
python -m scripts.build_index
```

Los PDF personales están excluidos del repositorio vía `.gitignore`;
`ejemplo_cv.md` se incluye para que la demo funcione sin configuración.
