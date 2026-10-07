export function generationOutputs(prompt) {
  const loaders = Object.entries(prompt).filter(([, node]) => node.class_type === "H3LVUnified");
  if (loaders.length !== 1) throw new Error("当前执行图需要一个 H3 长视频节点。");
  const loaderId = loaders[0][0];
  function usesLoader(id, visited = new Set()) {
    id = String(id);
    if (id === loaderId) return true;
    if (visited.has(id)) return false;
    visited.add(id);
    return Object.values(prompt[id]?.inputs || {}).some(input =>
      Array.isArray(input) && input.length === 2 && usesLoader(input[0], visited));
  }
  const outputs = Object.entries(prompt).filter(([, node]) => {
    let images;
    if (node.class_type === "VHS_VideoCombine") images = node.inputs?.images;
    else if (node.class_type === "SaveVideo") {
      const video = node.inputs?.video;
      const creator = Array.isArray(video) ? prompt[String(video[0])] : null;
      if (creator?.class_type === "CreateVideo") images = creator.inputs?.images;
    }
    return Array.isArray(images) && images.length === 2 && usesLoader(images[0]);
  });
  if (!outputs.length) {
    throw new Error("未找到连接到 H3 长视频的输出。请使用 VHS Video Combine，或“创建视频 → 保存视频”。");
  }
  return {loaderId, outputs};
}
