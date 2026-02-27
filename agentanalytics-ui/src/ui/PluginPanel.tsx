import React, { useEffect, useState } from "react";
import { Card, Alert, Button, Divider, List, Modal, Typography } from "antd";
import ReactMarkdown from "react-markdown";
import { artifactUrl } from "../lib/api";

async function fetchText(url: string) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return await r.text();
}

type PluginArtifact = {
  kind: string;
  relpath: string;
  description?: string;
};

export default function PluginPanel({
  runId,
  plugin,
  displayName
}: {
  runId: string;
  plugin: any;
  displayName: string;
}) {
  const artifacts: PluginArtifact[] = plugin?.artifacts || [];
  const summaryMd: string = plugin?.summary_md || "";

  // Prefer markdown report artifact
  const report = artifacts.find(
    (a) => a.kind === "markdown" || String(a.relpath).toLowerCase().endsWith(".md")
  );
  const reportUrl = report ? artifactUrl(runId, report.relpath) : null;

  const otherArtifacts = artifacts.filter((a) => a !== report);

  const [reportText, setReportText] = useState<string | null>(null);

  const [modalOpen, setModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState("");
  const [modalBody, setModalBody] = useState<string>("");
  const [modalErr, setModalErr] = useState<string | null>(null);

  const pluginStatus = plugin.status || "OK";
  const pluginError = plugin.error || null;

  // Load report markdown
  useEffect(() => {
    if (!reportUrl) {
      setReportText(null);
      return;
    }
    setReportText(null);
    fetchText(reportUrl)
      .then(setReportText)
      .catch(() => setReportText("Failed to load report."));
  }, [reportUrl]);

  async function openArtifact(a: PluginArtifact) {
    setModalErr(null);
    setModalTitle(a.description || a.relpath);
    setModalBody("Loading…");
    setModalOpen(true);
    try {
      const url = artifactUrl(runId, a.relpath);
      const text = await fetchText(url);

      // Pretty print JSON if applicable
      if (a.kind === "json" || String(a.relpath).toLowerCase().endsWith(".json")) {
        try {
          const obj = JSON.parse(text);
          setModalBody(JSON.stringify(obj, null, 2));
          return;
        } catch {
          // fallthrough to raw text
        }
      }

      setModalBody(text);
    } catch (e: any) {
      setModalErr(e?.message || "Failed to load artifact");
      setModalBody("");
    }
  }

  return (
    <div style={{ display: "grid", gap: 12 }}>
      {/* Summary card (Markdown) */}
      {pluginStatus === "FAILED" && (
          <Alert
            style={{ marginTop: 10 }}
            type="error"
            message="Plugin failed"
            description={
              pluginError
                ? `${pluginError.type || "Error"}: ${pluginError.message || ""}`
                : "Unknown error"
            }
          />
        )}
      <Card size="small">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <div>
            <Typography.Text strong style={{ fontSize: 16 }}>
              {displayName}
            </Typography.Text>{" "}
            <span style={{ color: "#888" }}>v{plugin.version}</span>
            <div style={{ color: "#666", marginTop: 4 }}>
              Ran in <b>{Number(plugin.elapsed_sec || 0).toFixed(2)}s</b>
            </div>
          </div>

          <div style={{ maxWidth: 720 }}>
            {summaryMd ? (
              <div style={{ whiteSpace: "normal" }}>
                <ReactMarkdown>{summaryMd}</ReactMarkdown>
              </div>
            ) : (
              <Typography.Text type="secondary">No summary.</Typography.Text>
            )}
          </div>
        </div>
      </Card>

      {/* Main report inline */}
      <Card size="small" title={report?.description || "Report"}>
        {!reportUrl ? (
          <Alert type="warning" message="No markdown report artifact found for this plugin." />
        ) : reportText == null ? (
          <Typography.Text type="secondary">Loading…</Typography.Text>
        ) : (
          <div style={{ maxWidth: 980 }}>
            <ReactMarkdown>{reportText}</ReactMarkdown>
          </div>
        )}
      </Card>

      {/* Other artifacts hidden behind links */}
      {otherArtifacts.length > 0 && (
        <Card size="small" title="Other artifacts">
          <List
            dataSource={otherArtifacts}
            renderItem={(a) => (
              <List.Item
                actions={[
                  <Button key="open" onClick={() => openArtifact(a)}>
                    Open
                  </Button>,
                  <Button
                    key="dl"
                    href={artifactUrl(runId, a.relpath)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download
                  </Button>
                ]}
              >
                <List.Item.Meta
                  title={a.description || a.relpath}
                  description={<span style={{ color: "#888" }}>{a.relpath}</span>}
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      <Modal
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={() => setModalOpen(false)}
        title={modalTitle}
        width={900}
      >
        {modalErr ? (
          <Alert type="error" message="Failed to load" description={modalErr} />
        ) : (
          <pre style={{ margin: 0, maxHeight: 540, overflow: "auto" }}>{modalBody}</pre>
        )}
      </Modal>
    </div>
  );
}