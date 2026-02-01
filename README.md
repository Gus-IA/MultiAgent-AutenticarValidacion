# 🧠 Business Idea Validator Agent (LangGraph + LangChain)

Este proyecto implementa un **agente conversacional multi-paso** para validar ideas de negocio utilizando **LangGraph**, **LangChain** y **OpenAI**, con búsqueda automática de perfiles relevantes en **LinkedIn** y soporte de *Human-in-the-Loop*.

El objetivo es ayudar a emprendedores a identificar **potenciales clientes reales** y generar mensajes de contacto y preguntas de validación.

---

## 🚀 ¿Qué hace el agente?

A partir de una **idea de negocio**, el sistema:

1. **Analiza la idea**
   - Identifica industrias relevantes
   - Roles y cargos clave
   - Tipos de empresas potencialmente interesadas
   - Expertise valioso para validar la idea

2. **Genera búsquedas estratégicas en LinkedIn**
   - Ejemplo:  
     - `restaurant industry CEO`
     - `fintech startup founder`
     - `healthcare technology director`

3. **Busca perfiles reales en LinkedIn**
   - Utiliza Tavily Search
   - Devuelve perfiles individuales y de empresas

4. **Crea mensajes personalizados**
   - Mensajes de introducción adaptados al perfil
   - 3 preguntas clave para validar la idea

5. **Incluye Human-in-the-Loop**
   - El agente puede pausar la ejecución y pedir ayuda humana
   - Ideal para validaciones, feedback o decisiones críticas

---

## 🧩 Arquitectura del sistema

El flujo está implementado con **LangGraph**:

```mermaid
START --> agent1 --> agent2 --> tools --> agent2 --> END


🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
