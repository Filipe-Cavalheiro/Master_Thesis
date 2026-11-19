# Notas do orador — defesa de mestrado (objetivo ≈ 19 min, limite 20 min)

Cada bloco `#### [Cena.k]` é um sub-slide (uma tecla), com 2–3 tópicos a falar. O `slides.py` lê
estes blocos ao renderizar e anexa-os como notas do orador do manim-slides (tecla `S` na exportação
HTML abre a vista do orador). Manter o formato dos títulos ao editar o texto.

## Orçamento de tempo

| # | Cena | Sub-slides | Tempo | Acumulado |
|---|---|---|---|---|
| 1 | TitleCard | 1 | 0:25 | 0:25 |
| 2 | Motivation | 5 | 1:50 | 2:15 |
| 3 | Problem | 2 | 0:55 | 3:10 |
| 4 | Roadmap | 1 | 0:35 | 3:45 |
| 5 | StateOfTheArt | 1 | 0:45 | 4:30 |
| 6 | Background | 4 | 1:30 | 6:00 |
| 7 | CentralizedDistribution | 2 | 1:05 | 7:05 |
| 8 | CentralizedControl | 3 | 1:50 | 8:55 |
| 9 | CentralizedResults | 4 | 2:20 | 11:15 |
| 10 | GeneralizedMesh | 5 | 1:30 | 12:45 |
| 11 | GeneralizedMPC | 4 | 2:00 | 14:45 |
| 12 | Results | 12 | 3:10 | 17:55 |
| 13 | Conclusions | 1 | 0:55 | 18:50 |
| 14 | FutureWork | 1 | 0:30 | 19:20 |
| 15 | ThankYou | 1 | 0:10 | 19:30 |
| — | Extra (só nas questões) | 8 | — | — |

≈ 19:30 de fala, abaixo do limite de 20 min. A etapa descentralizada na esfera
(cap. 5, cena `Decentralized`) está fora da apresentação; a etapa generalizada (cap. 6), também
descentralizada, é apresentada como "Descentralizado". Os vídeos dos testes reais passam a fase
principal em tempo real e o resto (descolagem, aproximação, regresso, aterragem) a 3x: 36 s e
44 s. Se estiver atrasado, avançar os vídeos com a tecla de seguinte assim que a ideia estiver
dita, ou encurtar a análise quantitativa dos Resultados e a ponderação pela velocidade (uma frase
cada).

Pontos de controlo: Roadmap concluído por ~3:45, Centralizado por ~11:15, Resultados por ~17:55.

---

## 1. TitleCard (0:25)

#### [TitleCard.1] Título
- Apresentação: Filipe Cavalheiro, orientação Prof. Bruno Guerreiro
- Ideia: um enxame circunda um drone intruso em 3D, em segurança e sem o destruir

## 2. Motivação (1:50)

#### [Motivation.1] Notícias
- Drones ao alcance de qualquer pessoa
- Drone perto de aeroporto → operações param (Bruxelas, Bélgica, Suécia)

#### [Motivation.2] Números da DFS
- Alemanha: 161 perturbações em 2024, 225 em 2025
- Voos atrasados/desviados; heliportos hospitalares
- Muitas incursões não intencionais: o operador não conhece a zona

#### [Motivation.3] Espaço aéreo restrito sobre Lisboa
- Zonas ANAC: vermelho = proibida (corredores das pistas), laranja = Área 1 (altura limitada)
- Atravessam a cidade: Santa Maria, Cidade Universitária, estádios, Oriente

#### [Motivation.4] Porque não as contramedidas habituais
- Jamming/spoofing: drone sem controlo + interfere com o aeroporto
- Interceção (redes, intercetores): captura ou colisão → destroços sobre a cidade
- Logo: conter a distância segura, não derrubar

#### [Motivation.5] Pergunta
- Circundar o intruso de forma segura, visível e não destrutiva
- Para o operador perceber que tem de sair

## 3. Problema (0:55)

#### [Problem.1] Objetivo 3D
- Alvo não cooperativo: só medição com ruído; cada perseguidor → p_i*
- Requisitos: 3D, alvo móvel, espaçamento uniforme, sem colisões
- Idealmente descentralizado e robusto à perda de agentes

#### [Problem.2] Âmbito
- Fora do âmbito: operadores maliciosos, captura física, procedimento legal

## 4. Roadmap (0:35)

#### [Roadmap.1] Duas etapas
- Três perguntas: para onde vai cada agente, como chega, como fica seguro
- Centralizado: Fibonacci + Lyapunov + APF
- Descentralizado: Lloyd em qualquer malha + MPC de Bernstein; estimativa do alvo por EMA + consenso

## 5. Estado da arte (0:45)

#### [StateOfTheArt.1] Abordagens
- Perseguição cíclica, líder–seguidor, campos de potencial: vantagens e falhas
- Lacuna: quase tudo planar; o 3D usa setores predefinidos

## 6. Fundamentos (1:30)

#### [Background.1] Algoritmo de Lloyd
- Células de Voronoi → mover sítio para o centroide → repetir
- Converge para CVT: distribuição uniforme (base da etapa descentralizada)

#### [Background.2] Vizinhos de Delaunay
- Dual do Voronoi: vizinhos partilham aresta
- Grafo de comunicação que se adapta ao movimento e fica conexo

#### [Background.3] Consenso
- Cada agente faz a média da estimativa do alvo com os vizinhos
- Estimativas convergem sem nó central

#### [Background.4] Métodos de segurança
- APF: atua na referência, sem garantia (cap. 4)
- CBF-QP: invariância, mas na entrada instantânea (não usado)
- MPC: restringe posições previstas + limites v, a; CBF-MPC é trabalho futuro

## 7. Centralizado — distribuição (1:05)

#### [CentralizedDistribution.1] Rede de Fibonacci
- Centralizado: um nó conhece todos → formação calculada diretamente
- Ângulo de ouro (~137,5°): pontos nunca se alinham, espaçamento uniforme

#### [CentralizedDistribution.2] Comparação
- vs lat–lon (só tamanhos de grelha) e disco de Poisson (CV 0,25)
- Fibonacci: CV 0,0105, menor energia de Riesz, ~700x mais rápido (0,1 ms)
- Única restrição: número ímpar de pontos

## 8. Centralizado — controlo (1:50)

#### [CentralizedControl.1] Decomposição
- Posição relativa = distância r_i × direção q_i na esfera
- Dois subobjetivos desacoplados: r_i → r*, q_i → q_i*

#### [CentralizedControl.2] Função de erro
- Erro clássico: norma sen θ → volta a zero no antípoda (falsa convergência)
- Erro de Ramp et al.: norma sen(θ/2), monótono

#### [CentralizedControl.3] Leis de controlo e teorema
- Geométrico roda q; radial com ganho exp(−|e_q|²): primeiro alinhar, depois aproximar
- Lyapunov + LaSalle: estabilidade quase global (exceto o antípoda)
- Erro radial exponencial, taxa ≥ k_r/e; animação: agente quase no antípoda converge

## 9. Centralizado — segurança e resultados (2:00)

#### [CentralizedResults.1] Salvaguarda APF
- Repulsão tangencial abaixo de d_min, reprojetada na esfera (só na referência)
- Esquerda sem, direita com; atribuição por Munkres (geodésica total mínima)

#### [CentralizedResults.2] Arquitetura completa
- Nó central: Fibonacci + Munkres
- Agente: geométrico + APF → radial → referência → Mellinger
- Alvo: medição com ruído → EMA → consenso Delaunay

#### [CentralizedResults.3] 31 agentes, alvo móvel
- Formação constitui-se e segue o alvo (ruído 5 %)
- Oscilação limitada em r* e erro ~1 m: não usa a velocidade do alvo
- Distância entre agentes sempre acima do mínimo

#### [CentralizedResults.4] Teste real
- 3 Crazyflies + OptiTrack, alvo virtual (magenta) em circunferência
- Verde real, laranja simulação: ~10 cm RMS
- Fases fora do controlador a 3x; tecla avança

## — Descentralizado na esfera (cap. 5): fora da apresentação (cena `Decentralized` fora de `scenes.txt`)

#### [Decentralized.1] Lloyd local na esfera
- Fibonacci precisa de N, Munkres de todas as posições
- Cada agente: Voronoi com vizinhos → centroide (Lloyd geodésico)

#### [Decentralized.2] Perda de agentes
- 4 perdidos; os 8 restantes redistribuem-se sem replaneamento global

#### [Decentralized.3] Invólucro convexo vs semiplano
- Convexo: rápido (0,03 s) mas pode criar células degeneradas
- Semiplano: mais uniforme (CV 0,073), mais lento (1,37 s), 93/100

#### [Decentralized.4] Arquitetura completa
- Igual, mas sem nó central: Lloyd com vizinhos Delaunay substitui Fibonacci + Munkres

#### [Decentralized.5] Resultados
- 12 agentes, 4 perdidos aos 10 s; aproximação lenta com alvo móvel
- CV < 0,1; distância mínima ~1,7 m, sem quase-colisões

## 10. Descentralizado — malhas e controlador de malha (1:30)

#### [GeneralizedMesh.1] Qualquer malha
- Sem nó central: Lloyd só com vizinhos Delaunay
- Malha como grafo pesado → Dijkstra → Lloyd em qualquer malha (busto, sapo)

#### [GeneralizedMesh.2] Controlador de malha
- Longe da fronteira: disco plano = barreira/aviso
- Interpolação disco–cone–esfera pela distância com sinal do alvo à fronteira
- Cores: nenhum vértice passa a fronteira

#### [GeneralizedMesh.3] Disco para cone
- Alvo atravessa → cone completo a meio raio para lá da fronteira

#### [GeneralizedMesh.4] Cone para esfera
- Mais dentro → esfera (envolvimento) a um raio; depois segue o alvo

#### [GeneralizedMesh.5] O alvo sai
- Processo inverte-se → volta à barreira

## 11. Descentralizado — MPC e velocidade (2:00)

#### [GeneralizedMPC.1] MPC de Bernstein
- Custo: distância ao objetivo de Lloyd + aceleração; limites v, a; envelope elipsoidal com vizinhos
- Polinómio de Bernstein grau 5: derivadas lineares; aplica 1.º passo e resolve de novo
- Nota: colisão é penalização, não restrição rígida

#### [GeneralizedMPC.2] Sem velocidade: CVT uniforme
- 12 pontos uniformes (Voronoi por Dijkstra)
- Alinhamento ~0, metade à frente do alvo

#### [GeneralizedMPC.3] Com velocidade: CVT ponderada
- Peso cresce com o ângulo à direção de trás → células da frente encolhem
- σ = 1,2: alinhamento ~0,32, 8/12 à frente

#### [GeneralizedMPC.4] Arquitetura completa
- Estimação do alvo igual; controlador de malha deforma a malha
- Lloyd ponderado substitui Fibonacci + Munkres; MPC substitui geométrico + radial → Mellinger

## 12. Resultados (3:10)

#### [Results.1] Repetição: disco na fronteira
- Cenário cap. 6 a 2x: 12 agentes, alvo em circunferência entra/sai de x > 0
- 1 s: disco na fronteira

#### [Results.2] Cone
- 2,5 s: alvo a atravessar → cone

#### [Results.3] Esfera fechada
- 4 s: alvo dentro → esfera fechada

#### [Results.4] Perda de agentes
- 10 s: 4 sem energia, 8 redistribuem-se

#### [Results.5] Regresso à barreira
- 13 s: alvo sai → barreira plana

#### [Results.6] Fim da simulação
- Deixar correr até ao fim

#### [Results.7] Análise quantitativa
- Distância mínima 3–4 m (> 1 m); velocidade sempre ≤ 3 m/s; recupera após a perda
- Fraquezas: agente perto do alvo (2–5 s, sem restrição agente–alvo)
- Erro ao objetivo até ~4 m, em escada (Voronoi a 1 Hz)

#### [Results.8] Teste real
- 4 Crazyflies: CVT na malha (azul) + MPC, alvo virtual magenta
- Real e simulado coincidem a poucos cm
- CF02 cai no regresso (anel vermelho = última posição); fases fora da formação a 3x

#### [Results.9] Aeroporto de Lisboa: barreira na parede
- Zona proibida real (ANAC), modelo X500, a 3x
- Barreira 30 m fora da zona; 13,9 s: alvo atravessa a parede → esfera

#### [Results.10] Perda de perseguidores
- 25 s: 4 sem energia, 8 redistribuem-se

#### [Results.11] O alvo entra na zona proibida
- 34,1 s: formação achata-se em barreira na fronteira

#### [Results.12] Barreira na fronteira
- A 1 m da fronteira até ao fim: nunca entram na zona

## 13. Conclusões (0:55)

#### [Conclusions.1] Conclusões
- Centralizado: simples, 31 agentes, convergência quase global
- Descentralizado: Lloyd em qualquer malha, barreira e envolvimento, MPC, robusto à perda
- Limitações: interior, alvo virtual, comunicação ideal, colisões só penalizadas

## 14. Trabalho futuro (0:30)

#### [FutureWork.1] Trabalho futuro
- Hardware real: PX4 + ROS 2, comunicação e perceção realistas
- Método: velocidade do alvo no MPC, alcançabilidade, CBF-MPC

## 15. Obrigado (0:10)

#### [ThankYou.1] Obrigado
- Agradecer; abrir para questões

#### [ThankYou.2] Questões (ciclo)
- (Animação em ciclo durante as questões)

---

## Slides extra (apenas durante as questões)

#### [BackupLyapunov.1] Esboço da prova
- V̇ ≤ 0; subníveis D_c (c < 2) compactos, invariantes, sem antípoda → LaSalle
- Radial em cascata, taxa ≥ k_r/e

#### [BackupControllers.1] Comparação de controladores
- fmincon: nunca chega a 0,1 m; APF: 4,46 s mas precisa de N, sem garantia
- Lyapunov: 17,6 s, erro quase nulo, menor computação (7,8 s)

#### [BackupDirect.1] Métodos de cálculo direto
- Lat–lon só para grelhas; Poisson CV 0,25; Fibonacci CV 0,011, menor Riesz

#### [BackupDirect.2] Tempo de execução
- Fibonacci ~0,1 ms, ~700x mais rápido que Poisson

#### [BackupIterative.1] Métodos de cálculo iterativo
- Thomson nunca converge; Lloyd convexo sempre (~20 it.)
- Lloyd semiplano: mais uniforme (CV 0,073), 93/100

#### [BackupIterative.2] Tempo de execução
- Semiplano ~40x mais lento → ambos testados no sistema completo

#### [BackupDecentralized.1] Falha do invólucro convexo
- Pode escolher o arco curto em vez do longo → partição errada, sítios colapsam
- Semiplanos evitam isto

#### [BackupParameters.1] Parâmetros da simulação descentralizada
- Parâmetros do cap. 6 (tabela da tese)
