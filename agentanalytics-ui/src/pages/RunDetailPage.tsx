import React, { useEffect, useMemo, useState } from "react";
import { Card, Tabs, Tag, Typography, Alert, Button, Table } from "antd";
import { useParams } from "react-router-dom";
import { artifactUrl, getRun, logsSseUrl } from "../lib/api";
import PluginPanel from "../ui/PluginPanel";
import { listPlugins } from "../lib/api";
import ReactMarkdown from "react-markdown";

function fmtTime(ms?: number) {
  if (!ms) return "-";
  return new Date(ms).toLocaleString();
}

export default function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [data, setData] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [pluginTitles, setPluginTitles] = useState<Record<string, string>>({});

  useEffect(() => {
    (async () => {
      try {
        const metas = await listPlugins();
        const m: Record<string, string> = {};
        for (const p of metas) m[p.name] = p.title;
        setPluginTitles(m);
      } catch {}
    })();
  }, []);

  async function refresh() {
    if (!runId) return;
    setLoading(true);
    try {
      const r = await getRun(runId);
      setData(r);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, [runId]);

  // SSE logs
  useEffect(() => {
    if (!runId) return;
    const es = new EventSource(logsSseUrl(runId));
    es.onmessage = (ev) => {
      setLogs((prev) => [...prev, ev.data].slice(-1200));
    };
    es.onerror = () => {};
    return () => es.close();
  }, [runId]);

  const manifest = data?.manifest || null;
  const status = data?.status || null;

  const statusStr = status?.status || "UNKNOWN";
  const statusColor =
    statusStr === "FINISHED" ? "green" : statusStr === "RUNNING" ? "blue" : "default";

  const plugins = manifest?.plugins || [];

  // Summary view of all plugins before tabs
  const pluginSummaryRows = plugins.map((p: any) => ({
    key: p.name,
    name: pluginTitles[p.name] || p.name,
    version: p.version,
    elapsed_sec: p.elapsed_sec,
    summary: p.summary || {},
    summary_md: p.summary_md || ""
  }));

  const showLogsTab = statusStr !== "FINISHED" || logs.length > 0;

  const [activeTopTab, setActiveTopTab] = useState<string>("overview");
  const [autoSwitchedToLogs, setAutoSwitchedToLogs] = useState<boolean>(false);

  const tabs = [
    {
      key: "overview",
      label: "Overview",
      children: (
        <>
          {!manifest ? (
            <Alert type="warning" message="No manifest yet (run may still be executing)" />
          ) : (
            <>
              <Typography.Paragraph>
                Started: <b>{fmtTime(manifest.started_ms)}</b><br />
                Finished: <b>{fmtTime(manifest.finished_ms)}</b><br />
                Trace count: <b>{manifest.trace_count ?? "-"}</b>
              </Typography.Paragraph>

              <Card size="small" title="Plugins summary" style={{ marginTop: 12 }}>
                <Table
                  rowKey="key"
                  dataSource={pluginSummaryRows}
                  pagination={false}
                  columns={[
                    { title: "Plugin", dataIndex: "name" },
                    {
                      title: "Version",
                      dataIndex: "version",
                      width: 120,
                      render: (v: string) => <span style={{ color: "#888" }}>{v}</span>
                    },
                    {
                      title: "Time",
                      dataIndex: "elapsed_sec",
                      width: 120,
                      render: (v: number) => `${Number(v || 0).toFixed(2)}s`
                    },
                    {
                      title: "Summary",
                      dataIndex: "summary_md",
                      render: (md: string) =>
                        md ? (
                          <div style={{ maxWidth: 760 }}>
                            <ReactMarkdown>{md}</ReactMarkdown>
                          </div>
                        ) : (
                          <span style={{ color: "#888" }}>-</span>
                        )
                    }
                  ]}
                />
              </Card>
            </>
          )}
        </>
      )
    },
    {
      key: "plugins",
      label: "Plugins",
      children: (
        !manifest ? (
          <Alert type="warning" message="No plugin results yet" />
        ) : (
          <Tabs
            tabPosition="left"
            items={plugins.map((p: any) => ({
              key: p.name,
              label: pluginTitles[p.name] || p.name,
              children: <PluginPanel runId={runId!} plugin={p} displayName={pluginTitles[p.name] || p.name} />
            }))}
          />
        )
      )
    }
  ];

  if (showLogsTab) {
    tabs.push({
      key: "logs",
      label: "Logs",
      children: (
        <div
          style={{
            background: "#0b0f14",
            color: "#d6deeb",
            padding: 12,
            borderRadius: 8,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
            fontSize: 12,
            height: 520,
            overflow: "auto",
            whiteSpace: "pre-wrap"
          }}
        >
          {logs.length ? logs.join("\n") : "No logs yet..."}
        </div>
      )
    });
  }

  useEffect(() => {
    if (!showLogsTab) return;

    const running = statusStr === "RUNNING" || statusStr === "QUEUED";
    if (running && !autoSwitchedToLogs) {
      setActiveTopTab("logs");
      setAutoSwitchedToLogs(true);
    }
  }, [statusStr, showLogsTab, autoSwitchedToLogs]);

  return (
    <Card
      title={
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <Typography.Text strong>Run {runId}</Typography.Text>
          <Tag color={statusColor}>{statusStr}</Tag>
        </div>
      }
      extra={<Button onClick={refresh} loading={loading}>Refresh</Button>}
    >
      <Tabs activeKey={activeTopTab} onChange={setActiveTopTab} items={tabs} />
    </Card>
  );
}