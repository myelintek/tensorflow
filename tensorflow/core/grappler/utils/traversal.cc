/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {
namespace grappler {

void ReverseDfs(const GraphView& graph_view, const std::vector<NodeDef*>& from,
                const std::function<void(NodeDef*)>& pre_order,
                const std::function<void(NodeDef*)>& post_order,
                const std::function<void(NodeDef*, NodeDef*)>& on_back_edge) {
  // Queue of work to do.
  struct QueueElem {
    NodeDef* node;
    bool children_visited;
    NodeDef* src;
  };
  std::queue<QueueElem> queue;

  for (NodeDef* node : from) {
    queue.push(QueueElem{node, false});
  }

  enum NodeState { NOT_VISITED = 0, VISITING = 1, DONE = 2 };
  std::unordered_map<NodeDef*, NodeState> node_state;
  while (!queue.empty()) {
    QueueElem w = queue.front();
    queue.pop();

    if (w.children_visited) {
      // We've processed all the children of this node
      node_state[w.node] = DONE;
      if (post_order) {
        post_order(w.node);
      }
      continue;
    }

    auto& rslt = node_state[w.node];
    if (rslt == DONE) {
      continue;
    } else if (rslt == VISITING) {
      // Loop detected
      if (on_back_edge) {
        on_back_edge(w.src, w.node);
      }
      continue;
    }
    rslt = VISITING;
    if (pre_order) {
      pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    queue.push(QueueElem{w.node, true, w.src});

    // Now enqueu the node children.
    for (const auto fanin : graph_view.GetFanins(*w.node, true)) {
      queue.push(QueueElem{fanin.node, false, w.node});
    }
  }
}

void ReverseBfs(const GraphView& graph_view, const std::vector<NodeDef*>& from,
                const std::function<void(NodeDef*)>& pre_order,
                const std::function<void(NodeDef*)>& post_order,
                const std::function<void(NodeDef*, NodeDef*)>& on_back_edge) {
  // Stack of work to do.
  struct StackElem {
    NodeDef* node;
    bool children_visited;
    NodeDef* src;
  };
  std::vector<StackElem> stack;

  stack.reserve(from.size());
  for (NodeDef* node : from) {
    stack.push_back(StackElem{node, false});
  }

  enum NodeState { NOT_VISITED = 0, VISITING = 1, DONE = 2 };
  std::unordered_map<NodeDef*, NodeState> node_state;
  while (!stack.empty()) {
    StackElem w = stack.back();
    stack.pop_back();

    if (w.children_visited) {
      // We've processed all the children of this node
      node_state[w.node] = DONE;
      if (post_order) {
        post_order(w.node);
      }
      continue;
    }

    auto& rslt = node_state[w.node];
    if (rslt == DONE) {
      continue;
    } else if (rslt == VISITING) {
      // Loop detected
      if (on_back_edge) {
        on_back_edge(w.src, w.node);
      }
      continue;
    }
    rslt = VISITING;
    if (pre_order) {
      pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.push_back(StackElem{w.node, true, w.src});

    // Now enqueu the node children.
    for (const auto fanin : graph_view.GetFanins(*w.node, true)) {
      stack.push_back(StackElem{fanin.node, false, w.node});
    }
  }
}

void ReverseDfsV2(const GraphView& graph_view, const std::vector<NodeDef*>& from,
                const NodeDef* to,
                const std::function<void(NodeDef*)>& pre_order,
                const std::function<void(NodeDef*)>& post_order,
                const std::function<void(NodeDef*, NodeDef*)>& on_back_edge) {
  // Stack of work to do.
  struct StackElem {
    NodeDef* node;
    bool children_visited;
    NodeDef* src;
  };
  std::vector<StackElem> stack;

  stack.reserve(from.size());
  for (NodeDef* node : from) {
    stack.push_back(StackElem{node, false, nullptr});
  }

  enum NodeState { NOT_VISITED = 0, VISITING = 1, DONE = 2 };
  std::unordered_map<NodeDef*, NodeState> node_state;
  while (!stack.empty()) {
    StackElem w = stack.back();
    stack.pop_back();

    // Don't go through the path contains destination
    if(w.node->name() == to->name())
      continue;

    if (w.children_visited) {
      // We've processed all the children of this node
      node_state[w.node] = DONE;
      if (post_order) {
        post_order(w.node);
      }
      continue;
    }

    auto& rslt = node_state[w.node];
    if (rslt == DONE) {
      continue;
    } else if (rslt == VISITING) {
      // Loop detected
      if (on_back_edge) {
        on_back_edge(w.src, w.node);
      }
      continue;
    }
    rslt = VISITING;
    if (pre_order) {
      pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.push_back(StackElem{w.node, true, w.src});

    // Now enqueu the node children.
    for (const auto fanin : graph_view.GetFanins(*w.node, true)) {
      stack.push_back(StackElem{fanin.node, false, w.node});
    }
  }
}

std::unordered_map<NodeDef*, NodeDef*> ReverseDfsV3(const GraphView& graph_view, const std::vector<NodeDef*>& from,
                const NodeDef* to,
                const std::function<void(NodeDef*)>& pre_order,
                const std::function<void(NodeDef*)>& post_order,
                const std::function<void(NodeDef*, NodeDef*)>& on_back_edge) {
  // Stack of work to do.
  struct StackElem {
    NodeDef* node;
    bool children_visited;
    NodeDef* src;
  };
  std::vector<StackElem> stack;
  std::unordered_map<NodeDef*, NodeDef*> tracing_path_map;
  
  stack.reserve(from.size());
  for (NodeDef* node : from) {
    stack.push_back(StackElem{node, false});
    tracing_path_map.insert({node, nullptr});
  }

  enum NodeState { NOT_VISITED = 0, VISITING = 1, DONE = 2 };
  std::unordered_map<NodeDef*, NodeState> node_state;
  while (!stack.empty()) {
    StackElem w = stack.back();
    stack.pop_back();

    // Don't go through the path contains destination
    if(w.node->name() == to->name()) {
      continue;
    }

    if (w.children_visited) {
      // We've processed all the children of this node
      node_state[w.node] = DONE;
      if (post_order) {
        post_order(w.node);
      }
      continue;
    }

    auto& rslt = node_state[w.node];
    if (rslt == DONE) {
      continue;
    } else if (rslt == VISITING) {
      // Loop detected
      if (on_back_edge) {
        on_back_edge(w.src, w.node);
      }
      continue;
    }
    rslt = VISITING;
    if (pre_order) {
      pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    stack.push_back(StackElem{w.node, true, w.src});

    // Now enqueu the node children.
    for (const auto fanin : graph_view.GetFanins(*w.node, true)) {
      if((w.node->name().find("gradients") != std::string::npos) &&
        (fanin.node->name() == to->name())) {
          continue;
      } else {
        stack.push_back(StackElem{fanin.node, false, w.node});
        // it is for walking the path from the f2 to f1.
        tracing_path_map.insert({fanin.node, w.node});
      }
    }
  }
  return tracing_path_map;
}

void BfsV2(const GraphView& graph_view, const std::vector<NodeDef*>& from,
                const NodeDef* to,
                const std::function<void(NodeDef*)>& pre_order,
                const std::function<void(NodeDef*)>& post_order,
                const std::function<void(NodeDef*, NodeDef*)>& on_back_edge) {
  // Queue of work to do.
  struct QueueElem {
    NodeDef* node;
    bool children_visited;
    NodeDef* src;
  };
  std::queue<QueueElem> queue;

  for (NodeDef* node : from) {
    queue.push(QueueElem{node, false});
  }

  enum NodeState { NOT_VISITED = 0, VISITING = 1, DONE = 2 };
  std::unordered_map<NodeDef*, NodeState> node_state;
  while (!queue.empty()) {
    QueueElem w = queue.front();
    queue.pop();

    // Don't go through the path contains destination
    if(w.node->name() == to->name())
      continue;

    if (w.children_visited) {
      // We've processed all the children of this node
      node_state[w.node] = DONE;
      if (post_order) {
        post_order(w.node);
      }
      continue;
    }

    auto& rslt = node_state[w.node];
    if (rslt == DONE) {
      continue;
    } else if (rslt == VISITING) {
      // Loop detected
      if (on_back_edge) {
        on_back_edge(w.src, w.node);
      }
      continue;
    }
    rslt = VISITING;
    if (pre_order) {
      pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    queue.push(QueueElem{w.node, true, w.src});

    // Now enqueu the node children.
    for (const auto fanout : graph_view.GetFanouts(*w.node, true)) {
      queue.push(QueueElem{fanout.node, false, w.node});
    }
  }
}

void ReverseBfsV2(const GraphView& graph_view, const std::vector<NodeDef*>& from,
                const NodeDef* to,
                const std::function<void(NodeDef*)>& pre_order,
                const std::function<void(NodeDef*)>& post_order,
                const std::function<void(NodeDef*, NodeDef*)>& on_back_edge) {
  // Queue of work to do.
  struct QueueElem {
    NodeDef* node;
    bool children_visited;
    NodeDef* src;
  };
  std::queue<QueueElem> queue;

  for (NodeDef* node : from) {
    queue.push(QueueElem{node, false});
  }

  enum NodeState { NOT_VISITED = 0, VISITING = 1, DONE = 2 };
  std::unordered_map<NodeDef*, NodeState> node_state;
  while (!queue.empty()) {
    QueueElem w = queue.front();
    queue.pop();

    // Don't go through the path contains destination
    if(w.node->name() == to->name())
      continue;

    if (w.children_visited) {
      // We've processed all the children of this node
      node_state[w.node] = DONE;
      if (post_order) {
        post_order(w.node);
      }
      continue;
    }

    auto& rslt = node_state[w.node];
    if (rslt == DONE) {
      continue;
    } else if (rslt == VISITING) {
      // Loop detected
      if (on_back_edge) {
        on_back_edge(w.src, w.node);
      }
      continue;
    }
    rslt = VISITING;
    if (pre_order) {
      pre_order(w.node);
    }

    // Enqueue the node again with the children_visited flag set to true.
    queue.push(QueueElem{w.node, true, w.src});

    // Now enqueu the node children.
    for (const auto fanin : graph_view.GetFanins(*w.node, true)) {
      queue.push(QueueElem{fanin.node, false, w.node});
    }
  }
}

}  // namespace grappler
}  // namespace tensorflow

