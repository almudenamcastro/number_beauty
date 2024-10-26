import streamlit as st
import streamlit.components.v1 as components
from streamlit_shap import st_shap
from lib.utils import Explainer

def app():
   # initialise the number
   if 'number' not in st.session_state:
        st.session_state['number'] = None
   # initialise the explainer
   explainer = Explainer()
   # page title
   st.markdown("# Velaske, ¿es bonito mi número?")

   # Number input box
   st.number_input('Número de lotería', 
                    0, 99999, 
                    value = None, 
                    key='number',
                    placeholder = "Escribe un número entre 0 y 99999 para medir su belleza",
                    )
   number = st.session_state['number']

   st.page_link(st.Page(about, title='¿Qué es esto?'), label='¿Qué es esto?', icon="💡")
   
   if number != None:
        st.divider()
        st.title(number)
        # load data 
        st.markdown('### '+ explainer.beauty_rating(number))
        # print plot
        st.markdown(explainer.features_explain(number))
        st_shap(explainer.features_plot(number))
#        summary = get_summary(data, number)
#        st.write(summary)


def about():
    # read article.md file
    
    with open('article.md') as f:
        st.markdown(f.read())

def main():
    st.set_page_config(
        page_title="beautifulnumbers", page_icon="🖼️"
        )    
    st.markdown('#### Números bonitos')
    pg = st.navigation({'Números bonitos': [st.Page(app, title="Comprueba tu número"), st.Page('article.py', title='¿Qué es esto?')]})
    pg.run()

if __name__ == '__main__':
    main()