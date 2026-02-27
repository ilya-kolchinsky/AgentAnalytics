import React, { useEffect, useState } from "react";
import { Card, Table, Tag, Typography, Descriptions } from "antd";
import { listPlugins, PluginMeta } from "../lib/api";

function renderDefaults(defaults: Record<string, any>) {
  const entries = Object.entries(defaults || {});
  if (!entries.length) return <Typography.Text type="secondary">None</Typography.Text>;

  return (
    <Descriptions size="small" column={1} bordered>
      {entries.map(([k, v]) => (
        <Descriptions.Item key={k} label={k}>
          {typeof v === "boolean" ? (v ? <Tag color="green">true</Tag> : <Tag>false</Tag>) : String(v)}
        </Descriptions.Item>
      ))}
    </Descriptions>
  );
}

function renderSettingsTags(p: PluginMeta) {
  const params = p.params || [];
  if (!params.length) return "-";

  // show keys; you can also include defaults in tooltip-ish fashion later
  return params.map((x) => (
    <Tag key={x.key} style={{ marginBottom: 4 }}>
      {x.key}
    </Tag>
  ));
}

export default function PluginsPage() {
  const [plugins, setPlugins] = useState<PluginMeta[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const p = await listPlugins();
        setPlugins(p);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <Card title="Supported Plugins">
      <Table
        rowKey="name"
        loading={loading}
        dataSource={plugins}
        pagination={{ pageSize: 20 }}
        expandable={{
          expandedRowRender: (r: PluginMeta) => (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div>
                <Typography.Text strong>Defaults</Typography.Text>
                <div style={{ marginTop: 8 }}>{renderDefaults(r.defaults || {})}</div>
              </div>
              <div>
                <Typography.Text strong>Parameters</Typography.Text>
                <Table
                  size="small"
                  rowKey="key"
                  pagination={false}
                  dataSource={r.params || []}
                  columns={[
                    { title: "Key", dataIndex: "key" },
                    { title: "Type", dataIndex: "type", width: 120 },
                    {
                      title: "Default",
                      dataIndex: "default",
                      width: 140,
                      render: (v: any) =>
                        typeof v === "boolean" ? (v ? <Tag color="green">true</Tag> : <Tag>false</Tag>) : (v ?? "-")
                    },
                    {
                      title: "Description",
                      dataIndex: "description",
                      render: (v: string) => <span style={{ color: "#666" }}>{v || "-"}</span>
                    }
                  ]}
                />
              </div>
            </div>
          )
        }}
        columns={[
          {
            title: "Plugin",
            dataIndex: "title",
            render: (_: any, r: PluginMeta) => (
              <div>
                <Typography.Text strong>{r.title}</Typography.Text>
                <div style={{ color: "#777" }}>{r.description}</div>
                <div style={{ color: "#aaa", fontSize: 12 }}>
                  {r.name} v{r.version}
                </div>
              </div>
            )
          },
          {
            title: "Requires",
            dataIndex: "requires",
            width: 280,
            render: (req: string[]) => (req?.length ? req.map((x) => <Tag key={x}>{x}</Tag>) : "-")
          },
          {
            title: "Settings",
            key: "settings",
            width: 420,
            render: (_: any, r: PluginMeta) => renderSettingsTags(r)
          }
        ]}
      />
    </Card>
  );
}