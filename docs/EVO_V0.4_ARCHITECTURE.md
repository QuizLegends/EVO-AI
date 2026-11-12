# EVO AI v0.4 — Arquitetura de Aprendizagem

## 1. Princípios

- Preservar o modelo neural e os componentes existentes.
- Separar treinamento, memória, avaliação e gerenciamento curricular.
- Não considerar redução de loss como comprovação de competência.
- Não permitir aprovação automática sem avaliação independente.
- Manter alterações controladas e versões estáveis.
- Separar dados de treinamento, validação e avaliação final.

## 2. Componentes

### Núcleo neural
Responsável pelo processamento e geração de respostas.

### Motor de aprendizagem
Responsável pelo treinamento e atualização dos pesos.

### Memória persistente
Armazena interações, exemplos e conhecimentos registrados.
Conteúdo não verificado não pode ser tratado como conhecimento aprovado.

### Avaliador neural
Acompanha loss, perplexidade e outras métricas do modelo.

### Avaliador de competências
Aplica exercícios inéditos e verifica objetivamente as respostas.

### Gerenciador curricular
Controla as etapas, os critérios de aprovação e a progressão.

### Planejador de melhorias
Analisa resultados e propõe alterações sem modificar automaticamente
a versão estável.

### Interface
Permite interação com a EVO e consulta ao progresso.

## 3. Primeira competência oficial

Identificar variáveis em códigos Python.

### Critérios

1. Identificar nomes de variáveis em atribuições simples.
2. Diferenciar nomes de variáveis dos valores armazenados.
3. Reconhecer ocorrências da mesma variável.
4. Diferenciar variáveis de palavras dentro de strings e comentários.
5. Identificar variáveis em atribuições múltiplas.

### Avaliação

- Primeira bateria: 25 exercícios inéditos.
- Segunda bateria: 25 exercícios independentes.
- Aprovação: 100% de acertos nas duas baterias.
- Questões de avaliação não podem integrar o treinamento.
- Respostas devem ser verificadas por um avaliador independente.

### Limites iniciais

Ficam fora desta primeira competência:

- Parâmetros de funções.
- Variáveis de laços.
- Classes e atributos.
- Escopo global e local.
- Compreensões.
- Expressões avançadas de atribuição.

Esses assuntos serão introduzidos em etapas posteriores.

## 4. Regra de progressão

A EVO não poderá avançar de competência sem cumprir todos os critérios
da etapa atual.

O resultado da avaliação deve ser registrado com:
- Identificação da competência.
- Versão da bateria de testes.
- Quantidade de acertos e erros.
- Critérios aprovados e reprovados.
- Data da avaliação.
- Resultado final.

## 5. Segurança da evolução

- Preservar checkpoints estáveis.
- Testar alterações em ambiente isolado.
- Executar a bateria de testes antes de aprovar mudanças.
- Não substituir versões estáveis sem validação.
- Manter backups permanentes.

## 6. Observação

100% de aprovação significa domínio comprovado dentro dos critérios
e das baterias definidas para a competência. Não representa garantia
de domínio absoluto de toda a linguagem Python.
