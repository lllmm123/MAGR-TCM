import streamlit as st
import json
import os
import base64
import random
import streamlit.components.v1 as components
from pyvis.network import Network

# 导入后端逻辑 (确保 medical_agents.py 在同级目录)
from medical_agents import (
    初始化系统, global_agent1, global_agent3, 
    运行妇科病情分析器, 运行完整检索流程, 内容评估与重试系统_优化赋分版
)

# ==========================================
# 1. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="中医妇科智能决策系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 尽早尝试初始化后台知识引擎，防止空指针报错
try:
    初始化系统()
except Exception as e:
    pass

# ==========================================
# 2. 用户数据库与路径搜索辅助函数
# ==========================================
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

def find_all_paths(kg, start_nid, end_nid, max_hops=3):
    """在知识图谱中寻找两个实体间的所有路径"""
    paths = []
    stack = [(start_nid, [start_nid], [])] 
    while stack:
        (current, path, rels) = stack.pop()
        if len(path) > max_hops: continue
        for neighbor in kg.get_neighbors(current):
            next_node = neighbor['target']
            rel = neighbor['rel']
            if next_node == end_nid:
                paths.append((path + [next_node], rels + [rel]))
            elif next_node not in path:
                stack.append((next_node, path + [next_node], rels + [rel]))
    return paths

# ==========================================
# 3. 初始化会话状态 (Session State)
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_sessions" not in st.session_state:
    st.session_state.history_sessions = {}
if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = "新对话"

# ==========================================
# 4. 背景图片处理与毛玻璃特效 (终极防遮挡)
# ==========================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def apply_custom_style(is_login_page=True):
    try:
        bin_str = get_base64_of_bin_file('background.jpg')
        
        common_style = f'''
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.65), rgba(255, 255, 255, 0.65)), 
                        url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        [data-testid="stChatMessage"], .stExpander, div[data-testid="stStatusWidget"], .welcome-card {{
            background-color: rgba(255, 255, 255, 0.75) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
        }}
        
        p, li, h1, h2, h3, h4, span {{
            text-shadow: 0px 1px 2px rgba(255, 255, 255, 0.9);
        }}
        
        .prescription-box {{
            border: 2px solid #1a3a3a; padding: 25px; 
            background: rgba(255, 255, 255, 0.85); 
            backdrop-filter: blur(15px);
            font-family: "KaiTi", "楷体", serif; position: relative; 
            box-shadow: 6px 6px 0px rgba(26,58,58,0.8);
            border-radius: 8px;
        }}
        .rx-stamp {{
            font-size: 50px; font-weight: bold; color: #d9534f;
            position: absolute; top: 10px; left: 20px; opacity: 0.8;
        }}
        </style>
        '''
        st.markdown(common_style, unsafe_allow_html=True)
    except:
        pass

    if is_login_page:
        st.markdown('''
        <style>
        .main .block-container {
            max-width: 800px; margin: 80px auto;
            background-color: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 20px; padding: 40px 60px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid rgba(255, 255, 255, 0.6);
        }
        </style>
        ''', unsafe_allow_html=True)

# ==========================================
# 5. 智能图谱渲染引擎 (极致散开防重叠 + 低视角清晰频繁漫游版)
# ==========================================
TYPE_COLOR = {
    "疾病": "#E53E3E", "证型": "#DD6B9C", "处方": "#DD6B20", 
    "药物": "#38A169", "症状": "#3182CE", "治法": "#805AD5", 
    "体格检查": "#008080", "脉象": "#8B4513", "舌象": "#FF69B4", 
    "辅助检查结果": "#00CED1", "面色": "#D2B48C"
}

def get_node_type(node):
    if not node: return "证型"
    if "labels" in node and isinstance(node["labels"], list) and len(node["labels"]) > 0:
        label = node["labels"][0].strip()
        if label in TYPE_COLOR: return label
    
    name = node.get("name", "").strip().lower()
    if "脉" in name: return "脉象"
    if "舌" in name or "苔" in name: return "舌象"
    if "面色" in name or "面" in name or "唇" in name: return "面色"
    if any(word in name for word in ["B超", "常规", "阴性", "阳性", "超声", "检查", "心电图", "指标"]): return "辅助检查结果"
    if any(word in name for word in ["压痛", "包块", "触诊", "宫颈光滑", "附件"]): return "体格检查"

    zhengxing_keywords = ["气滞", "血瘀", "郁滞", "湿热", "痰湿", "肾虚", "脾虚", "肝郁", "不足", "痰火", "阴虚", "阳虚", "寒湿", "气虚", "血虚", "漏证", "滑胎", "脏躁"]
    if any(word in name for word in zhengxing_keywords): return "证型"
    zhifa_keywords = ["逐瘀", "祛湿", "止痛", "疏肝", "理气", "活血", "养血", "健脾", "化痰", "清心", "宁神", "益气", "滋阴", "温阳", "止血", "解毒"]
    if any(word in name for word in zhifa_keywords): return "治法"
    if any(word in name for word in ["汤", "丸", "散", "加减", "饮", "膏"]): return "处方"
    jibing_keywords = ["不调", "痛经", "闭经", "不孕", "崩漏", "癥瘕", "带下病", "子肿", "乳癖", "产后"]
    if any(word in name for word in jibing_keywords): return "疾病"
    zhengzhuang_keywords = ["痛", "酸", "胀", "热", "汗", "晕", "干", "乏力", "瘙痒", "恶心", "失眠", "口苦"]
    if any(word in name for word in zhengzhuang_keywords): return "症状"
    return "药物"

def draw(kg_instance, nodes, edges, auto_pan=False, height="750px"):
    if kg_instance is None: return
    nodes = [n for n in nodes if n]
    edges = [e for e in edges if e[0] in nodes and e[1] in nodes]
    
    net = Network(height=height, width="100%", bgcolor="rgba(255,255,255,0.5)", directed=True)
    
    for nid in nodes:
        node = kg_instance.get_node(nid)
        t = get_node_type(node)
        full_name = node.get("name", f"实体_{nid}")
        display_name = full_name if len(full_name) <= 4 else full_name[:4] + "..."
        net.add_node(nid, label=display_name, title=full_name, color=TYPE_COLOR[t], shape="circle")
    
    for s, t, r in edges:
        net.add_edge(s, t, label=r)

    # 👑 修改点：字号加大到18，加入描边，解决“字迹糊”问题
    neo4j_options = """
    {
      "nodes": { 
          "borderWidth": 1.5, 
          "color": { "border": "#ffffff" }, 
          "shadow": {"enabled": true, "color": "rgba(0,0,0,0.15)", "size": 8, "x": 2, "y": 2},
          "font": { 
              "color": "#ffffff", 
              "size": 18, 
              "face": "Microsoft YaHei",
              "strokeWidth": 2,
              "strokeColor": "#333333"
          }, 
          "margin": 10 
      },
      "edges": { 
          "arrows": {"to": {"enabled": true, "scaleFactor": 0.65}}, 
          "color": { "color": "#7BD1E1", "highlight": "#FF9800", "inherit": false }, 
          "font": { "size": 12, "align": "horizontal", "background": "#ffffff", "strokeWidth": 0 }, 
          "smooth": false 
      },
      "physics": { 
          "barnesHut": { 
              "gravitationalConstant": -8000, 
              "springLength": 200, 
              "springConstant": 0.04,
              "avoidOverlap": 1 
          }, 
          "solver": "barnesHut", 
          "stabilization": {"iterations": 300} 
      },
      "interaction": { 
          "hover": false, 
          "tooltipDelay": 200, 
          "selectConnectedEdges": false 
      }
    }
    """
    net.set_options(neo4j_options)
    net.save_graph("tmp.html")
    
    with open("tmp.html", "r", encoding="utf-8") as f:
        html_content = f.read()
        
    if auto_pan:
        # 👑 修改点：加快移动频率，调低高度拉近视角
        cinematic_js = """
        <script type="text/javascript">
            network.on("stabilizationIterationsDone", function () {
                network.setOptions( { physics: { enabled: false } } );
                
                var nodeIds = network.body.nodeIndices;
                if (nodeIds.length === 0) return;
                
                var validNodes = nodeIds.filter(id => network.getConnectedEdges(id).length > 1);
                if (validNodes.length === 0) validNodes = nodeIds;
                
                function smoothRoam() {
                    var targetId = validNodes[Math.floor(Math.random() * validNodes.length)];
                    var edgeCount = network.getConnectedEdges(targetId).length;

                    var targetScale;

                    if (edgeCount > 8) {
                        // 视角压得更近
                        targetScale = 1.0;
                    } else if (edgeCount > 3) {
                        targetScale = 1.3;
                    } else {
                        targetScale = 1.6;
                    }
                    
                    network.focus(targetId, { 
                        scale: targetScale, 
                        animation: { 
                            // 移动速度变快（2500 -> 1500）
                            duration: 1500, 
                            easingFunction: 'easeInOutCubic' 
                        } 
                    });
                }
                
                setTimeout(function() { 
                    smoothRoam(); 
                    // 移动频率变高（3500 -> 2000）
                    setInterval(smoothRoam, 2000); 
                }, 500);
            });
        </script>
        </body>
        """
        html_content = html_content.replace("</body>", cinematic_js)
        
    components.html(html_content, height=int(height.replace("px", "")) + 20)

# ==========================================
# 6. 登录与注册页
# ==========================================
def login_register_page():
    st.markdown("<h1 style='text-align: center;'>🌿 中医妇科智能决策系统</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>请先登录或注册</h4>", unsafe_allow_html=True)
    st.write("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔑 用户登录", "📝 新用户注册"])
        users_db = load_users()
        with tab_login:
            login_user = st.text_input("用户名", key="login_username")
            login_pwd = st.text_input("密码", type="password", key="login_password")
            if st.button("立即登录", use_container_width=True):
                if login_user in users_db and users_db[login_user] == login_pwd:
                    st.session_state.logged_in = True
                    st.session_state.username = login_user
                    st.rerun()
                else:
                    st.error("用户名或密码错误")
        with tab_register:
            reg_user = st.text_input("设置用户名", key="reg_username")
            reg_pwd = st.text_input("设置密码", type="password", key="reg_password")
            reg_pwd_confirm = st.text_input("确认密码", type="password", key="reg_password_confirm")
            if st.button("注册账号", use_container_width=True):
                if reg_user and reg_pwd == reg_pwd_confirm:
                    users_db[reg_user] = reg_pwd
                    save_users(users_db)
                    st.success("注册成功！请登录")
                else:
                    st.error("输入有误或密码不一致")

# ==========================================
# 7. 主界面流程
# ==========================================
if not st.session_state.logged_in:
    apply_custom_style(is_login_page=True)
    login_register_page()
    st.stop()
else:
    apply_custom_style(is_login_page=False)
    
    with st.sidebar:
        st.markdown("<div style='text-align: center; margin-bottom: 10px;'><span style='font-size: 60px;'>🌿</span></div>", unsafe_allow_html=True)
        st.write(f"👋 欢迎, **{st.session_state.username}**")
        if st.button("退出登录"):
            st.session_state.logged_in = False
            st.rerun()
        st.write("---")
        
        if st.button("➕ 开启新对话", use_container_width=True, type="primary"):
            if st.session_state.messages:
                st.session_state.history_sessions[st.session_state.current_session_name] = st.session_state.messages
            st.session_state.messages = []
            st.session_state.current_session_name = "新对话"
            st.session_state.pop("latest_roam_nodes", None)
            st.session_state.pop("latest_roam_edges", None)
            st.rerun()

        st.markdown("<div class='sidebar-header'>📜 历史记录</div>", unsafe_allow_html=True)
        for s_name in list(st.session_state.history_sessions.keys()):
            if st.button(f"💬 {s_name}", use_container_width=True, key=f"hist_{s_name}"):
                st.session_state.messages = st.session_state.history_sessions[s_name]
                st.session_state.current_session_name = s_name
                st.session_state.pop("latest_roam_nodes", None)
                st.session_state.pop("latest_roam_edges", None)
                st.rerun()
        
        st.write("---")
        status_placeholder = st.empty()
        status_placeholder.success("🟢 系统就绪")
        
        st.markdown("<div class='sidebar-header'>⚙️ 引擎配置</div>", unsafe_allow_html=True)
        st.markdown("- **领域图谱**: ✅ 已加载\n- **版本**: MAGR-TCM v9.2\n- **异构博弈**: 已开启")
        
        if st.button("🔄 清空当前会话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("latest_roam_nodes", None)
            st.session_state.pop("latest_roam_edges", None)
            st.rerun()

    st.markdown("<h1 style='color: #1A5237; font-weight: 600; text-align: center; margin-bottom: 0.5rem;'>中医妇科智能决策系统</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #5C836D; text-align: center; font-size: 1.1rem; margin-bottom: 2rem;'>基于多智能体(MDT)会诊与图谱逻辑寻踪的家庭健康分诊平台</p>", unsafe_allow_html=True)

    tab_chat, tab_graph = st.tabs(["💬 智能问诊会诊", "🕸️ 知识图谱探索舱"])

    with tab_chat:
        if not st.session_state.messages:
            st.markdown("<div class='welcome-card'><h4>💡 欢迎使用 MAGR-TCM 智能决策系统</h4><p>请在下方输入框用日常语言描述您的病情，系统将自动调度四个医疗智能体执行“感知-路由-推理-质控”流。</p></div>", unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        if prompt := st.chat_input("请在此详述主诉症状 (如: 月经推迟、肢体沉重等)..."):
            if st.session_state.current_session_name == "新对话":
                st.session_state.current_session_name = prompt[:12] + "..."
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status_placeholder.warning("🟡 智能体会诊中...")
                with st.status("🌿 正在启动多智能体(MDT)联合会诊...", expanded=True) as status_box:
                    try:
                        初始化系统()
                        st.write("🕵️ **Agent 1**: 临床语义级细粒度状态标定中...")
                        分析结果, 会话ID, Agent2数据 = 运行妇科病情分析器(prompt)
                        
                        if 分析结果:
                            st.write("🕸️ **Agent 2 & 3**: 动态弹性束搜索与图谱逻辑关联中...")
                            decision, 检索结果 = 运行完整检索流程(prompt, 分析结果)
                            
                            ragas_contexts = []
                            if decision['selected_source'] == 'knowledge_graph':
                                log = 检索结果.get('metadata', {}).get('trace_log', '')
                                ragas_contexts = [l for l in log.split('\n') if l.strip()]

                            st.write("⚕️ **Agent 4**: 异构双引擎触发辩论，严格阻断事实幻觉中...")
                            
                            import medical_agents
                            评估系统 = 内容评估与重试系统_优化赋分版()
                            最终结果 = 评估系统.评估并决策(prompt, 检索结果, 分析结果, ragas_contexts)
                            
                            status_box.update(label="✅ MDT 会诊决策完毕", state="complete", expanded=False)
                            status_placeholder.success("🟢 循证方案已生成")
                            
                            ans = 最终结果.get("final_content", [""])[0]
                            
                            formatted_ans = f"""<div class='prescription-box'>
                                <div class='rx-stamp'>℞</div>
                                <div style='text-align:center; border-bottom:2px solid #1a3a3a; margin-bottom:15px;'>
                                    <h3>中医家庭辅助分诊会诊单</h3><p style='color:gray;font-size:12px;'>会诊系统溯源码: {会话ID}</p>
                                </div>
                                {ans}
                            </div>"""
                            st.markdown(formatted_ans, unsafe_allow_html=True)
                            st.session_state.messages.append({"role": "assistant", "content": formatted_ans})

                            import medical_agents
                            kg_instance = medical_agents.global_agent3.kg
                            if kg_instance:
                                focus_ids = []
                                if 分析结果 and '症状' in 分析结果:
                                    for ent in 分析结果['症状']:
                                        name = ent.get('名称', '').replace('(疑似)','').strip()
                                        if name in kg_instance.inverted_index:
                                            focus_ids.extend(kg_instance.inverted_index[name])

                                if not focus_ids:
                                    valid_seeds = sorted([nid for nid in kg_instance.nodes if get_node_type(kg_instance.get_node(nid)) in ["疾病", "证型"]])
                                    focus_ids = [valid_seeds[0]] if valid_seeds else sorted(list(kg_instance.nodes.keys()))[:1]
                                else:
                                    focus_ids = sorted(list(set(focus_ids)))

                                visited = set(focus_ids[:3]) 
                                edges_to_draw = []
                                queue = list(visited)
                                queue.sort() 

                                while queue and len(visited) < 80: 
                                    cur = queue.pop(0)
                                    neighbors = kg_instance.get_neighbors(cur)
                                    neighbors.sort(key=lambda x: str(x.get('target', '')))
                                    
                                    for e in neighbors:
                                        tgt = e["target"]
                                        if tgt not in visited:
                                            visited.add(tgt)
                                            edges_to_draw.append((cur, tgt, e["rel"]))
                                            queue.append(tgt)
                                            if len(visited) >= 80: break

                                st.session_state.latest_roam_nodes = list(visited)
                                st.session_state.latest_roam_edges = edges_to_draw

                        else:
                            status_box.update(label="❌ 分析失败", state="error")
                    except Exception as e:
                        st.error(f"系统架构错误: {e}")

        if "latest_roam_nodes" in st.session_state and st.session_state.latest_roam_nodes:
            st.markdown("---")
            st.markdown("<h4 style='color:#1A5F2A; text-align:center;'>🌌 本次诊断的白盒化推理图谱溯源</h4>", unsafe_allow_html=True)
            st.markdown("<div style='background:rgba(255,255,255,0.7); backdrop-filter:blur(10px); border-radius:12px; padding:10px; border:1px solid rgba(255,255,255,0.8);'>", unsafe_allow_html=True)
            
            import medical_agents
            if medical_agents.global_agent3 and medical_agents.global_agent3.kg:
                draw(medical_agents.global_agent3.kg, st.session_state.latest_roam_nodes, st.session_state.latest_roam_edges, auto_pan=True, height="750px")
            
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_graph:
        st.markdown("<h3 style='color:#1A5F2A;'>🕸️ 中医妇科知识网络全量勘探舱</h3>", unsafe_allow_html=True)
        初始化系统()
        import medical_agents
        kg = medical_agents.global_agent3.kg

        with st.expander("🎨 系统实体类型映射字典", expanded=True):
            cols = st.columns(6)
            for i, (label, color) in enumerate(TYPE_COLOR.items()):
                col_idx = i % 6
                cols[col_idx].markdown(f"<div style='background:{color};color:white;padding:6px;margin-bottom:8px;text-align:center;border-radius:4px;font-size:14px;box-shadow:0 2px 4px rgba(0,0,0,0.1);'>{label}</div>", unsafe_allow_html=True)

        mode = st.radio("图谱探索模式", ["🔹 单体邻域关联提取算法 (BFS)", "🔸 双端逻辑隐性路径寻踪 (Multi-hop)"], horizontal=True, label_visibility="collapsed")

        if mode == "🔹 单体邻域关联提取算法 (BFS)":
            name = st.text_input("输入医学实体名称 (Anchor Entity)", "气滞血瘀")
            if name in kg.inverted_index:
                start_id = kg.inverted_index[name][0]
                def get_massive_nodes(sid, depth=3):
                    visited, edges, q = {sid}, [], [sid]
                    for _ in range(depth):
                        nxt = []
                        for cur in q:
                            for e in kg.get_neighbors(cur):
                                t = e["target"]
                                if t not in visited:
                                    visited.add(t); edges.append((cur, t, e["rel"])); nxt.append(t)
                        q = nxt
                        if not q: break
                    return list(visited), edges
                nodes, edges = get_massive_nodes(start_id)
                draw(kg, nodes, edges, height="850px")

        elif mode == "🔸 双端逻辑隐性路径寻踪 (Multi-hop)":
            c1, c2 = st.columns(2)
            a_n = c1.text_input("起始实体 (Source Node)", "气滞血瘀").strip()
            b_n = c2.text_input("目标实体 (Target Node)", "助阳逐瘀").strip()
            
            if st.button("🔍 触发路径探索算法", use_container_width=True):
                if a_n in kg.inverted_index and b_n in kg.inverted_index:
                    aid, bid = kg.inverted_index[a_n][0], kg.inverted_index[b_n][0]
                    with st.spinner(f"引擎正在计算【{a_n}】到【{b_n}】的隐性路径 (剪枝上限: 20 Hops)..."):
                        paths = find_all_paths(kg, aid, bid, 20)
                        if paths:
                            nodes, edges = set(), []
                            for p_n, p_r in paths:
                                nodes.update(p_n)
                                for i in range(len(p_r)): edges.append((p_n[i], p_n[i+1], p_r[i]))
                            st.success(f"✅ 探索完毕！底层图算法共提取出 {len(paths)} 条医疗逻辑关联路径。")
                            draw(kg, list(nodes), edges, height="850px")
                        else:
                            st.warning(f"⚠️ 在图谱中未发现【{a_n}】和【{b_n}】在 20 步之内的隐性关联。")
                else:
                    missing = []
                    if a_n not in kg.inverted_index: missing.append(f"起始实体: {a_n}")
                    if b_n not in kg.inverted_index: missing.append(f"目标实体: {b_n}")
                    st.error(f"❌ 实体未命中图谱索引库：{', '.join(missing)}")