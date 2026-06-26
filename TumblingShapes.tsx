import React, { useEffect, useRef } from 'react';
import ReactDOM from 'react-dom';
import Matter from 'matter-js';
var decomp = require('poly-decomp');

if (typeof window !== 'undefined') {
  Matter.Common.setDecomp(decomp);
}

interface Project {
  title: string;
  category?: string;
  src?: string;
  content?: React.ReactNode;
  type?: 'project' | 'link' | 'campaign' | 'image' | 'question';
  prompt?: string;
  shape?: 'letterH' | 'letterC' | 'letterR' | 'letterI' | 'letterS';
  imageSrc?: string;
  url?: string;
}

interface TumblingShapesProps {
  projects?: Project[];
  filterCategory?: string;
  onShapeClick?: (item: Project) => void;
  mode?: 'initial' | 'links';
  chatCentered?: boolean;
  containerId?: string;
}

const TumblingShapes: React.FC<TumblingShapesProps> = ({
  projects = [],
  filterCategory,
  onShapeClick,
  mode = 'initial',
  chatCentered = false,
  containerId = 'output_area'
}) => {
  const sceneRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<Matter.Engine | null>(null);
  const renderRef = useRef<Matter.Render | null>(null);
  const bodiesRef = useRef<{ body: Matter.Body; project: Project; shapeType?: string }[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hoveredBodyRef = useRef<Matter.Body | null>(null);

  useEffect(() => {
    if (!sceneRef.current) return;

    const Engine = Matter.Engine,
          Render = Matter.Render,
          Runner = Matter.Runner,
          Bodies = Matter.Bodies,
          Body = Matter.Body,
          Composite = Matter.Composite,
          Mouse = Matter.Mouse,
          MouseConstraint = Matter.MouseConstraint,
          Events = Matter.Events;

    const engine = Engine.create();
    engineRef.current = engine;
    const world = engine.world;

    const render = Render.create({
      element: sceneRef.current,
      engine,
      options: {
        width: window.innerWidth,
        height: window.innerHeight,
        wireframes: false,
        background: mode === 'initial' ? '#1f1409a1' : '#dcd3c3'
      }
    });
    renderRef.current = render;
    canvasRef.current = render.canvas;
    if (render.canvas) {
      render.canvas.style.position = 'relative';
      render.canvas.style.zIndex = '600';
    }

    function getHVertices() {
      const vertices: {x:number,y:number}[] = [];
      const stemWidth = 90;
      const totalHeight = 400;
      const archStartY = 230;
      const legWidth = 62;
      const radius = 80;
      vertices.push({ x: 0, y: totalHeight });
      vertices.push({ x: 0, y: -10 });
      vertices.push({ x: stemWidth, y: 0 });
      vertices.push({ x: stemWidth, y: archStartY });
      const archCx = stemWidth;
      const archCy = archStartY + radius;
      for (let i = 0; i <= 20; i++) {
        const angle = -Math.PI / 2 + (i / 20) * (Math.PI / 2);
        vertices.push({ x: archCx + radius * Math.cos(angle), y: archCy + radius * Math.sin(angle) });
      }
      const rightLegX = archCx + radius;
      vertices.push({ x: rightLegX + 8, y: totalHeight });
      vertices.push({ x: rightLegX - legWidth + 16, y: totalHeight });
      const innerRadius = radius - legWidth + 16;
      for (let i = 0; i <= 15; i++) {
        const angle = 0 - (i / 15) * (Math.PI / 2);
        vertices.push({ x: archCx + innerRadius * Math.cos(angle), y: archCy + innerRadius * Math.sin(angle) + 32 });
      }
      vertices.push({ x: stemWidth, y: totalHeight });
      return vertices;
    }
    function getCVertices() {
      const vertices: {x:number,y:number}[] = [];
      const radius = 176;
      const thickness = 152;
      const segments = 40;
      for (let i = 0; i <= segments; i++) {
        const angle = (Math.PI * 0.6) + (i / segments) * (Math.PI * 1.00);
        vertices.push({ x: Math.cos(angle) * radius, y: Math.sin(angle) * radius });
      }
      const innerRadius = radius - thickness;
      for (let i = segments; i >= 0; i--) {
        const angle = (Math.PI * 0.55) + (i / segments) * (Math.PI * 1.15);
        vertices.push({ x: Math.cos(angle) * innerRadius, y: Math.sin(angle) * innerRadius });
      }
      return vertices;
    }
    function getRVertices() {
      const vertices: {x:number,y:number}[] = [];
      const stemWidth = 96;
      const totalHeight = 332;
      const shoulderRadius = 294;
      const shoulderStartY = 198;
      vertices.push({ x: 0, y: totalHeight });
      vertices.push({ x: 16, y: 16 });
      vertices.push({ x: stemWidth, y: 0 });
      vertices.push({ x: stemWidth, y: shoulderStartY });
      const shoulderCx = stemWidth;
      const shoulderCy = shoulderStartY + shoulderRadius * 0.3;
      for (let i = 0; i <= 20; i++) {
        const angle = -Math.PI / 2 + (i / 20) * (Math.PI / 4.9);
        vertices.push({ x: shoulderCx + Math.cos(angle) * shoulderRadius * 0.5, y: shoulderCy + Math.sin(angle) * shoulderRadius * 0.9 });
      }
      const innerRadius = shoulderRadius * 0.55;
      for (let i = 20; i >= 0; i--) {
        const angle = -Math.PI / 2 + (i / 20) * (Math.PI / 6.9);
        vertices.push({ x: shoulderCx + Math.cos(angle) * innerRadius, y: shoulderCy + Math.sin(angle) * innerRadius + 16 });
      }
      vertices.push({ x: stemWidth, y: shoulderStartY + 32 });
      vertices.push({ x: stemWidth, y: totalHeight });
      return vertices;
    }
    function getIVertices() {
      const vertices: {x:number,y:number}[] = [];
      const width = 96;
      const height = 302;
      const dotRadius = 64;
      const dotGap = 4.6;
      vertices.push({ x: -width / 2, y: height / 2.1 });
      vertices.push({ x: -width / 2, y: -height / 2.1 });
      vertices.push({ x: -width / 2, y: -height / 2 - dotGap });
      const dotCenterY = -height / 2 - dotGap - dotRadius;
      const segments = 20;
      for (let i = 0; i <= segments; i++) {
        const angle = Math.PI + (i / segments) * Math.PI * 1.4;
        vertices.push({ x: Math.cos(angle) * dotRadius, y: dotCenterY + Math.sin(angle) * dotRadius });
      }
      vertices.push({ x: width / 2, y: -height / 2 - dotGap });
      vertices.push({ x: width / 2, y: -height / 2 });
      vertices.push({ x: width / 2, y: height / 2 });
      return vertices;
    }
    function getTopCVertices() {
      const vertices: {x:number,y:number}[] = [];
      const radius = 160;
      const thickness = 72;
      const segments = 40;
      for (let i = 0; i <= segments; i++) {
        const angle = (Math.PI * 0.65) + (i / segments) * (Math.PI * 1.0);
        vertices.push({ x: Math.cos(angle) * radius, y: Math.sin(angle) * radius });
      }
      const innerRadius = radius - thickness;
      for (let i = segments; i >= 0; i--) {
        const angle = (Math.PI * 0.55) + (i / segments) * (Math.PI * 1.15);
        vertices.push({ x: Math.cos(angle) * innerRadius, y: Math.sin(angle) * innerRadius });
      }
      return vertices;
    }
    function getBottomCVertices() {
      const vertices: {x:number,y:number}[] = [];
      const radius = 144;
      const thickness = 64;
      const segments = 25;
      for (let i = 0; i <= segments; i++) {
        const angle = (Math.PI * 1.65) + (i / segments) * (Math.PI * 1.3);
        vertices.push({ x: Math.cos(angle) * radius, y: Math.sin(angle) * radius });
      }
      const innerRadius = radius - thickness;
      for (let i = segments; i >= 0; i--) {
        const angle = (Math.PI * 1.65) + (i / segments) * (Math.PI * 1.0);
        vertices.push({ x: Math.cos(angle) * innerRadius, y: Math.sin(angle) * innerRadius });
      }
      return vertices;
    }

    const consciousnessQuestions = [
      "Is consciousness fundamentally physical?",
      "Can consciousness exist without a brain?",
      "What is the relationship between awareness and self?",
      "Is consciousness continuous or discrete?",
      "How does subjective experience arise?",
      "Does consciousness imply free will?",
      "Can machines ever be conscious?",
      "Is there a difference between human and animal consciousness?",
      "Does the universe itself have consciousness?",
      "Can consciousness be measured objectively?"
    ];

    const itemsToRender = projects.length > 0 ? projects : consciousnessQuestions.map(q => ({
      title: q,
      type: 'question' as const
    }));

    const projectBodies: Matter.Body[] = [];
    bodiesRef.current = [];

    itemsToRender.forEach((project, index) => {
      if (!project) return;
      let shapeType: string;
      const isImageType = project.type === 'image' && project.imageSrc;
      if (mode === 'initial') {
        switch (project.title) {
          case 'copywriting': shapeType = 'letterC'; break;
          case 'humanized ai': shapeType = 'letterH'; break;
          case 'resume scan tool': shapeType = 'letterR'; break;
          case 'interactive/ux': shapeType = 'letterI'; break;
          case 'simplified technical content': shapeType = 'letterS'; break;
          default: shapeType = ['letterC','letterH','letterR','letterI','letterS'][index % 5];
        }
      } else {
        if ((project as any).shape) {
          shapeType = (project as any).shape;
        } else if (project.title.length > 40) {
          shapeType = 'letterH';
        } else if (index < 5) {
          shapeType = ['letterC','letterH','letterR','letterI','letterS'][index];
        } else {
          shapeType = ['letterC','letterH','letterR','letterI','letterS'][Math.floor(Math.random()*5)];
        }
      }
      const spacing = window.innerWidth / (itemsToRender.length + 1);
      const x = spacing * (index + 1);
      const y = -200 - (Math.floor(index) * 700);
      let body: Matter.Body;
      let fillColor: string;
      let strokeColor: string;
      if (isImageType) {
        fillColor = 'rgba(0,0,0,0)';
        strokeColor = 'rgba(0,0,0,0)';
      } else if (mode === 'initial') {
        fillColor = '#dcd3c3';
        strokeColor = '#dcd3c3';
      } else {
        fillColor = '#312113';
        strokeColor = '#312113';
      }
      switch (shapeType) {
        case 'letterC':
          body = Bodies.fromVertices(x, y, [getCVertices()], { render: { fillStyle: fillColor, strokeStyle: strokeColor, lineWidth:1 }, restitution:0.1, friction:1, slop:.02 }, true);
          break;
        case 'letterH':
          body = Bodies.fromVertices(x, y, [getHVertices()], { render: { fillStyle: fillColor, strokeStyle: strokeColor, lineWidth:1 }, restitution:0.1, friction:50, slop:.02 }, true);
          break;
        case 'letterR':
          body = Bodies.fromVertices(x, y, [getRVertices()], { render: { fillStyle: fillColor, strokeStyle: strokeColor, lineWidth:1 }, restitution:0.1, friction:50, slop:.02 }, true);
          break;
        case 'letterI':
          body = Bodies.fromVertices(x, y, [getIVertices()], { render: { fillStyle: fillColor, strokeStyle: strokeColor, lineWidth:1 }, restitution:0.1, friction:50, slop:.02 }, true);
          break;
        default:
          body = Body.create({
            parts: [
              Bodies.fromVertices(x, y-80, [getTopCVertices()], { render:{fillStyle:fillColor} }),
              Bodies.fromVertices(x, y+80, [getBottomCVertices()], { render:{fillStyle:fillColor} })
            ]
          });
      }
      projectBodies.push(body);
      bodiesRef.current.push({ body, project, shapeType });
    });

    const ground = Bodies.rectangle(window.innerWidth/2, window.innerHeight, window.innerWidth, 60, { isStatic:true, render:{fillStyle:'#180f07'} });
    const leftWall = Bodies.rectangle(-30, window.innerHeight/2, 30, window.innerHeight*2, { isStatic:true, render:{fillStyle:'#1e1208'} });
    const rightWall = Bodies.rectangle(window.innerWidth+30, window.innerHeight/2, 30, window.innerHeight*2, { isStatic:true, render:{fillStyle:'#1a0e04'} });
    Composite.add(world, [...projectBodies, ground, leftWall, rightWall]);

    Render.run(render);
    const runner = Runner.create();
    Runner.run(runner, engine);
    const delta = 1000/160;
    engine.timing.timeScale = 0.5;
    let animationFrameId: number;
    const animate = () => { Engine.update(engine, delta); animationFrameId = requestAnimationFrame(animate); };
    animate();

    const imageCache = new Map<string, HTMLImageElement>();
    bodiesRef.current.forEach(({ project }) => {
      if ((project as any).imageSrc) {
        const img = new Image();
        img.src = (project as any).imageSrc;
        imageCache.set((project as any).imageSrc, img);
      }
    });

    Events.on(render, 'afterRender', () => {
      const context = render.context;
      const fontSize = mode === 'links' ? '18px' : '22px';
      context.font = `${fontSize} "kcgangster", Arial`;
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      bodiesRef.current.forEach(({ body, project, shapeType }) => {
        const { position, angle } = body;
        const isImageShape = (project as any).type === 'image' && (project as any).imageSrc;
        if (isImageShape) {
          const imageSrc = (project as any).imageSrc;
          const img = imageCache.get(imageSrc);
          if (img && img.complete) {
            context.save();
            context.translate(position.x, position.y);
            context.rotate(angle);
            if (mode === 'links') { context.rotate(-Math.PI/2); }
            const imgW = 170, imgH = 170;
            context.drawImage(img, -imgW/2, -imgH/2, imgW, imgH);
            context.restore();
          }
        } else {
          const isHovered = hoveredBodyRef.current?.id === body.id;
          context.fillStyle = mode === 'initial' ? (isHovered ? '#dcd3c3' : 'rgb(20,12,5)') : (isHovered ? 'rgb(26,17,8)' : '#dcd3c3');
          context.save();
          context.translate(position.x, position.y);
          context.rotate(angle);
          if (shapeType === 'letterH' || shapeType === 'letterR') context.rotate(-Math.PI/2);
          else if (shapeType === 'letterC') { context.rotate(3 * -Math.PI/1.58); context.rotate(Math.PI/2); }
          else if (shapeType === 'letterS') context.rotate(-Math.PI/2);
          else if (shapeType === 'letterI') context.rotate(3 * -Math.PI/2);
          else if (mode === 'links') context.rotate(-Math.PI/2);
          let displayTitle = project.title;
          if (mode === 'links') {
            const maxLength = 70;
            if (displayTitle.length > maxLength) displayTitle = displayTitle.substring(0, maxLength) + '...';
          }
          let xOffset = 0, yOffset = 0;
          if (mode === 'links') {
            if (shapeType === 'letterH') { xOffset = 20; yOffset = -12; }
            if (shapeType === 'letterR') { xOffset = -30; yOffset = -25; }
            if (shapeType === 'letterI') { xOffset = 50; yOffset = 15; }
          } else if (mode === 'initial') {
            if (shapeType === 'letterH' || shapeType === 'letterR') { xOffset = -30; yOffset = -32; }
          }
          context.fillText(displayTitle, xOffset, yOffset);
          context.restore();
        }
      });
    });

    const mouse = Mouse.create(render.canvas);
    const mouseConstraint = MouseConstraint.create(engine, { mouse, constraint:{ stiffness:0.2, render:{visible:false} } });
    Composite.add(world, mouseConstraint);
    const originalColors = new Map();
    bodiesRef.current.forEach(({ body, project }) => {
      const isImg = (project as any).type === 'image';
      let origFill = isImg ? 'rgba(0,0,0,0)' : (mode==='initial' ? '#dcd3c3' : '#2a1a0b');
      let origStroke = isImg ? 'rgba(0,0,0,0)' : (mode==='initial' ? '#dcc3' : '#27180c');
      originalColors.set(body.id,{fill:origFill,stroke:origStroke});
      if (body.parts && body.parts.length>1) body.parts.forEach(p=>{ if(p.id!==body.id) originalColors.set(p.id,{fill:origFill,stroke:origStroke});});
    });
    Events.on(mouseConstraint,'mousemove',event=>{
      const mp = event.mouse.position;
      let foundHover: Matter.Body | null = null;
      bodiesRef.current.forEach(({ body })=>{ if(Matter.Bounds.contains(body.bounds,mp)&&Matter.Vertices.contains(body.vertices,mp)) foundHover=body; });
      if(foundHover!==hoveredBodyRef.current){
        if(hoveredBodyRef.current){
          const orig = originalColors.get(hoveredBodyRef.current.id); if(orig){ hoveredBodyRef.current.render.fillStyle=orig.fill; hoveredBodyRef.current.render.strokeStyle=orig.stroke; if(hoveredBodyRef.current.parts) hoveredBodyRef.current.parts.forEach(p=>p.render.fillStyle=orig.fill); }
        }
        if(foundHover){
          const map = bodiesRef.current.find(m=>m.body===foundHover);
          const isImg = map && (map.project as any).type==='image';
          if(!isImg){ const hoverColor = mode==='initial' ? 'rgb(18,11,5)' : '#dcd3c3'; foundHover.render.fillStyle=hoverColor; foundHover.render.strokeStyle=hoverColor; if(foundHover.parts) foundHover.parts.forEach(p=>p.render.fillStyle=hoverColor); }
        }
        hoveredBodyRef.current = foundHover;
      }
    });
    Events.on(mouseConstraint,'mousedown',event=>{
      const mp = event.mouse.position;
      for(const bm of bodiesRef.current){ if(Matter.Bounds.contains(bm.body.bounds,mp)&&Matter.Vertices.contains(bm.body.vertices,mp)){ if(onShapeClick) onShapeClick(bm.project); break;} }
    });
    render.canvas.style.cursor='grab';
    Events.on(mouseConstraint,'startdrag',()=>{render.canvas.style.cursor='grabbing';});
    Events.on(mouseConstraint,'enddrag',()=>{render.canvas.style.cursor='grab';});

    return () => {
      if(animationFrameId) cancelAnimationFrame(animationFrameId);
      Render.stop(render);
      Runner.stop(runner);
      Composite.clear(world,false);
      Engine.clear(engine);
      render.canvas.remove();
      render.textures={};
    };
  },[projects,filterCategory,onShapeClick]);

  const containerEl = typeof window !== 'undefined'?document.getElementById(containerId):null;
  const content = <div ref={sceneRef} style={{width:'100%',height:'100%'}} />;
  if(containerEl){return ReactDOM.createPortal(content,containerEl);}return content;
};

export default TumblingShapes;
