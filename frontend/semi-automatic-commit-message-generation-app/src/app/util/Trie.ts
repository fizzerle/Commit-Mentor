export class trie_node {
  public terminal: boolean;
  public filename!: string;
  public path!: string;
  public children: Map<string, trie_node>;
  public selected: boolean = false;
  public mode: string = "";
  public parent!: trie_node;

  public getChildren(): trie_node[] {
    return Array.from(this.children.values())
  }
  constructor() {
    this.terminal = false;
    this.children = new Map();
  }
}

export class trie<T> {
  public root: trie_node;
  private elements: number;

  constructor() {
    this.root = new trie_node();
    this.elements = 0;
  }

  public get length(): number {
    return this.elements;
  }

  public get(key: string): string | null {
    const node = this.getNode(key);
    if (node) {
      return node.filename;
    }
    return null;
  }

  public contains(key: string): boolean {
    const node = this.getNode(key);
    return !!node;
  }

  public insert(key: string, value: string,mode?:string): void {
    let node = this.root;
    let remaining = key;
    while (remaining.length > 0) {
      // @ts-ignore
      let child: trie_node<T> = null;
      for (const childKey of node.children.keys()) {
        const prefix = this.commonPrefix(remaining, childKey);
        if (!prefix.length) {
          continue;
        }
        if (prefix.length === childKey.length) {
          // enter child node
          // @ts-ignore
          child = node.children.get(childKey);
          remaining = remaining.slice(childKey.length+1);
          break;
        } else {
          // split the child
          child = new trie_node();
          const child2 = node.children.get(childKey)
          if(child2){
            child2.path = childKey.slice(prefix.length+1);
            child2.parent = child
          }
          child.children.set(
            childKey.slice(prefix.length),
            // @ts-ignore
            node.children.get(childKey)
          );
          node.children.delete(childKey);
          child.path = prefix;
          child.parent = node
          child.mode = mode
          node.children.set(prefix, child);
          remaining = remaining.slice(prefix.length+1);
          break;
        }
      }
      if (!child && remaining.length) {
        child = new trie_node();
        child.path = remaining
        child.parent = node
        child.mode = mode
        node.children.set(remaining, child);
        remaining = "";
      }
      node = child;
    }
    if (!node.terminal) {
      node.terminal = true;
      this.elements += 1;
    }
    node.filename = remaining;
  }

  public remove(key: string): void {
    const node = this.getNode(key);
    if (node) {
      node.terminal = false;
      this.elements -= 1;
    }
  }

  private getNode(key: string): trie_node | null {
    let node = this.root;
    let remaining = key;
    while (node && remaining.length > 0) {
      let child = null;
      for (let i = 1; i <= remaining.length; i += 1) {
        child = node.children.get(remaining.slice(0, i));
        if (child) {
          remaining = remaining.slice(i);
          break;
        }
      }
      // @ts-ignore
      node = child;
    }
    return remaining.length === 0 && node && node.terminal ? node : null;
  }

  private commonPrefix(a: string, b: string): string {
    const asplit = a.split("/")
    const bsplit = b.split("/")
    const shortest = Math.min(asplit.length, bsplit.length);
    let i = 0;
    for (; i < shortest; i += 1) {
      if (asplit[i] !== bsplit[i]) {
        break;
      }
    }

    return (asplit.slice(0, i)).join("/");
  }
}
