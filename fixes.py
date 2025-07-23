# fixes.py - Arquivo com todas as correções necessárias

"""
INSTRUÇÕES DE IMPLEMENTAÇÃO:

1. CORREÇÃO DO ERRO save_config:
   - No arquivo app.rev2.py, linha ~93 (função test_api_key):
     Trocar: EnhancedSessionManager.save_api_key(api_key)
     Por: UserAwareSessionManager.save_api_key(api_key)

2. CORREÇÃO DA VERIFICAÇÃO DE API KEY DO .ENV:
   - Substituir o método get_client() da classe OpenAIService
   - Substituir o método show_api_config() da classe UIComponents
   - Usar as implementações abaixo
"""

import os
from dotenv import load_dotenv
import streamlit as st
from typing import Optional
import openai
from typing import Dict, List, Any, Optional, Tuple

# ========== CORREÇÃO 1: save_config ==========

def patch_user_database_manager():
    """
    Adiciona método save_config ao UserDatabaseManager para compatibilidade
    """
    from user_auth_system import UserDatabaseManager
    import json
    
    def save_config(self, key: str, value: Any):
        """Método de compatibilidade que redireciona para save_user_config"""
        if hasattr(st.session_state, 'current_user') and st.session_state.current_user:
            user_id = st.session_state.current_user.id
            self.save_user_config(user_id, key, value)
        else:
            # Para usuários não logados, salvar como 'anonymous'
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_config (user_id, key, value, updated_at)
                    VALUES ('anonymous', ?, ?, CURRENT_TIMESTAMP)
                """, (key, json.dumps(value) if not isinstance(value, str) else value))
    
    # Adicionar método à classe
    UserDatabaseManager.save_config = save_config


# ========== CORREÇÃO 2: API Key com .env ==========

class OpenAIServiceFixed:
    """Versão corrigida do OpenAIService com verificação de .env"""
    
    @staticmethod
    def get_client() -> Optional[openai.OpenAI]:
        """Obtém cliente OpenAI com priorização correta"""
        # Carregar .env
        load_dotenv()
        
        # Ordem de prioridade para obter API key
        api_key = None
        
        # 1. Verificar variável de ambiente primeiro
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            api_key = env_key
            # Salvar no session state para uso consistente
            st.session_state.openai_api_key = env_key
            
            # Se usuário logado, salvar para o usuário
            if hasattr(st.session_state, 'current_user') and st.session_state.current_user:
                from user_auth_system import UserAwareSessionManager
                UserAwareSessionManager.save_api_key(env_key)
        
        # 2. Se não encontrou no .env, verificar session state
        elif st.session_state.get('openai_api_key'):
            api_key = st.session_state.openai_api_key
        
        # 3. Se usuário logado, verificar configuração do usuário
        elif hasattr(st.session_state, 'current_user') and st.session_state.current_user:
            if hasattr(st.session_state, 'db_manager'):
                user_id = st.session_state.current_user.id
                user_key = st.session_state.db_manager.get_user_config(user_id, 'openai_api_key')
                if user_key:
                    api_key = user_key
        
        if not api_key:
            return None
        
        try:
            return openai.OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Erro ao configurar OpenAI: {str(e)}")
            return None


class UIComponentsFixed:
    """Versão corrigida do UIComponents"""
    
    @staticmethod
    def show_api_config():
        """Configuração da API com verificação de .env"""
        # Verificar se existe API key (incluindo .env)
        load_dotenv()
        env_key = os.getenv('OPENAI_API_KEY')
        session_key = st.session_state.get('openai_api_key')
        
        has_key = env_key or session_key
        
        if has_key:
            col1, col2 = st.columns([3, 1])
            with col1:
                if env_key:
                    st.success("✅ OpenAI API configurada via arquivo .env!")
                else:
                    st.success("✅ OpenAI API configurada!")
            with col2:
                if st.button("🔄 Reconfigurar"):
                    # Só remove do session state, não do .env
                    if 'openai_api_key' in st.session_state:
                        del st.session_state.openai_api_key
                    st.rerun()
        else:
            st.warning("⚠️ Configure sua chave da OpenAI para funcionalidades completas de IA")
            with st.expander("⚙️ Configurar OpenAI API"):
                st.info("""
                💡 **Dica:** Para configuração permanente, crie um arquivo `.env` na raiz do projeto com:
                ```
                OPENAI_API_KEY=sua_chave_aqui
                ```
                Ou configure manualmente abaixo:
                """)
                
                api_key = st.text_input(
                    "Chave da API OpenAI", 
                    type="password", 
                    help="Cole sua chave da API da OpenAI aqui"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Salvar Chave"):
                        if api_key:
                            # Testar a chave
                            try:
                                test_client = openai.OpenAI(api_key=api_key)
                                test_client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": "test"}],
                                    max_tokens=1
                                )
                                
                                # Salvar se válida
                                from user_auth_system import UserAwareSessionManager
                                UserAwareSessionManager.save_api_key(api_key)
                                st.success("✅ Chave salva e validada!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Chave inválida: {str(e)}")
                        else:
                            st.error("❌ Digite uma chave válida")
                
                with col2:
                    if st.button("🔗 Obter Chave OpenAI"):
                        st.info("Visite: https://platform.openai.com/api-keys")


# ========== FUNÇÃO PARA APLICAR TODAS AS CORREÇÕES ==========

def apply_all_fixes():
    """
    Aplica todas as correções necessárias.
    Chamar esta função no início do main() em app.rev2.py
    """
    # Aplicar patch no UserDatabaseManager
    patch_user_database_manager()
    
    # Substituir métodos nas classes originais
    from app import OpenAIService, UIComponents
    OpenAIService.get_client = OpenAIServiceFixed.get_client
    UIComponents.show_api_config = UIComponentsFixed.show_api_config
    
    print("✅ Todas as correções aplicadas com sucesso!")


# ========== EXEMPLO DE USO NO main() ==========
"""
# No início da função main() em app.rev2.py, adicionar:

from fixes import apply_all_fixes

def main():
    # Aplicar correções
    apply_all_fixes()
    
    # Resto do código...
    st.set_page_config(...)
"""