import React from "react";
import { Layout, Menu, Typography } from "antd";
import { Routes, Route, useNavigate, useLocation } from "react-router-dom";
import RunsPage from "./pages/RunsPage";
import RunDetailPage from "./pages/RunDetailPage";
import NewRunPage from "./pages/NewRunPage";
import PluginsPage from "./pages/PluginsPage";

const { Header, Sider, Content } = Layout;

export default function App() {
  const nav = useNavigate();
  const loc = useLocation();

  const selectedKey = loc.pathname.startsWith("/new")
    ? "new"
    : loc.pathname.startsWith("/plugins")
      ? "plugins"
      : "runs";

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider width={220} theme="light">
        <div style={{ padding: 16 }}>
          <Typography.Title level={4} style={{ margin: 0 }}>
            AgentAnalytics
          </Typography.Title>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            Insights from your MLFlow traces
          </Typography.Text>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          items={[
            { key: "runs", label: "Runs", onClick: () => nav("/runs") },
            { key: "new", label: "New Run", onClick: () => nav("/new") },
            { key: "plugins", label: "Plugins", onClick: () => nav("/plugins") }
          ]}
        />
      </Sider>

      <Layout>
        <Header style={{ background: "#fff", borderBottom: "1px solid #f0f0f0" }}>
          <Typography.Text>
            {selectedKey === "runs" ? "Runs" : selectedKey === "plugins" ? "Plugins" : "Create Run"}
          </Typography.Text>
        </Header>

        <Content style={{ padding: 20 }}>
          <Routes>
            <Route path="/" element={<RunsPage />} />
            <Route path="/runs" element={<RunsPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
            <Route path="/new" element={<NewRunPage />} />
            <Route path="/plugins" element={<PluginsPage />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}