import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require('C:/Users/Lucas/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try {
 const page=await browser.newPage({viewport:{width:1280,height:1000},deviceScaleFactor:1.5});
 const failures=[];page.on('pageerror',e=>failures.push(e.message));
 await page.goto(pathToFileURL(path.resolve('output/VLA_详细图解讲义.html')).href);
 await page.waitForFunction(()=>[...document.images].every(i=>i.complete));
 const metrics=await page.evaluate(()=>({
  formulas:document.querySelectorAll('img.math-inline').length,
  displays:document.querySelectorAll('.math-display').length,
  broken:[...document.images].filter(i=>i.naturalWidth===0).map(i=>i.alt),
  overflow:document.documentElement.scrollWidth>innerWidth,
  raw:[...document.querySelectorAll('p,blockquote')].filter(e=>/Σ_\(|L_actor|πθ|Qψ|a_\(t/.test(e.textContent)).map(e=>e.textContent)
 }));
 assert.ok(metrics.formulas>300);assert.equal(metrics.broken.length,0);assert.equal(metrics.raw.length,0);assert.equal(metrics.overflow,false);
 for (const [name,text] of [['policy','1.1 从分类器到闭环策略'],['tensor','2.1 张量、token、embedding'],['qkv','2.2 一次 attention 在做什么'],['rl','15.4 动作块级 critic 的 TD 目标']]) {
  await page.getByRole('heading',{name:text,exact:true}).scrollIntoViewIfNeeded();
  await page.screenshot({path:`tmp/pdfs/html-math-${name}.png`});
 }
 await page.setViewportSize({width:390,height:844});
 await page.getByRole('heading',{name:'15.4 动作块级 critic 的 TD 目标',exact:true}).scrollIntoViewIfNeeded();
 await page.screenshot({path:'tmp/pdfs/html-math-mobile.png'});
 console.log('Mobile bounds:',await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('main *')].filter(e=>e.getBoundingClientRect().right>innerWidth+2&&!e.closest('.tablewrap')).map(e=>({tag:e.tagName,text:e.textContent.slice(0,50),right:e.getBoundingClientRect().right})).slice(0,10)})));
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 assert.deepEqual(failures,[]);
 fs.writeFileSync('tmp/pdfs/math-checks.json',JSON.stringify(metrics,null,2));
 console.log(metrics);
} finally {await browser.close();}
