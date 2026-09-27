"""FastAPI REST server exposing `RetailProductDetector` over HTTP.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import BASE_DIR, OUTPUT_DIR, UPLOADS_DIR
from detector import DetectorInferenceError, DetectorInitError, RetailProductDetector
from pos_app.cart import ShoppingCart
from pos_app.config import (
    DATASET_IMAGES_DIR,
    MAX_SUGGESTIONS,
    POS_CONFIDENCE_THRESHOLD,
    SUGGESTION_MIN_CONFIDENCE,
)
from pos_app.dataset_utils import (
    get_catalog,
    get_product,
    next_capture_index,
    record_checkout_training_sample,
    register_product,
)
from pos_app.orders import save_order
from pos_app.pos_detector import load_checkout_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail Product Detector",
    description="YOLO11/YOLOv8-based retail product detection for supermarket/pharmacy shelves.",
    version="1.0.0",
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/product-images", StaticFiles(directory=str(DATASET_IMAGES_DIR)), name="product-images")

detector: RetailProductDetector | None = None

# Separate from `detector` above: this one is fine-tuned on the onboarded
# barcodes (when available) and powers the browser Checkout tab. Lazily
# loaded on first use, and reloaded after a successful training run so a
# freshly fine-tuned model is picked up without restarting the server.
checkout_detector = None
checkout_cart = ShoppingCart()
training_process: subprocess.Popen | None = None
training_log_path = BASE_DIR / "training_log.txt"


def _get_checkout_detector():
    global checkout_detector
    if checkout_detector is None:
        checkout_detector = load_checkout_detector()
    return checkout_detector


def _cart_payload() -> dict:
    items = [
        {
            "barcode": item.barcode,
            "name": item.name,
            "price": item.price,
            "quantity": item.quantity,
            "subtotal": round(item.price * item.quantity, 2),
        }
        for item in checkout_cart.items.values()
    ]
    return {"items": items, "total": round(checkout_cart.total(), 2)}


class CheckoutAddRequest(BaseModel):
    barcode: str
    record_training: bool = False
    crop_path: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the browser UI for the learning catalog and checkout flow."""
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Retail POS</title>
            <style>
                :root {
                    --bg: #08111f;
                    --panel: #101a2c;
                    --card: #16263d;
                    --muted: #9fb3c8;
                    --text: #ecf3ff;
                    --accent: #2dd4bf;
                    --accent2: #60a5fa;
                    --warning: #fbbf24;
                    --danger: #f87171;
                }
                * { box-sizing: border-box; }
                body {
                    margin: 0;
                    font-family: Arial, sans-serif;
                    background: linear-gradient(135deg, #020817 0%, #0f172a 100%);
                    color: var(--text);
                    min-height: 100vh;
                    padding: 26px;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .tabs {
                    display: flex;
                    gap: 12px;
                    margin-bottom: 18px;
                    flex-wrap: wrap;
                }
                .tab {
                    border: 1px solid rgba(159,179,200,0.2);
                    background: rgba(16,26,44,0.8);
                    color: var(--text);
                    padding: 12px 18px;
                    border-radius: 10px;
                    cursor: pointer;
                    font-weight: 700;
                }
                .tab.active {
                    background: linear-gradient(135deg, var(--accent), var(--accent2));
                    color: #04111e;
                    border-color: transparent;
                }
                .panel-grid {
                    display: grid;
                    grid-template-columns: 1.2fr 0.8fr;
                    gap: 22px;
                }
                .panel {
                    background: rgba(16,26,44,0.94);
                    border: 1px solid rgba(159,179,200,0.18);
                    border-radius: 18px;
                    padding: 22px;
                    box-shadow: 0 18px 40px rgba(2,8,23,0.45);
                }
                h1, h2, h3 { margin-top: 0; }
                form, .stack { display: grid; gap: 12px; }
                input, button, select {
                    width: 100%;
                    padding: 12px 14px;
                    border-radius: 10px;
                    border: 1px solid rgba(159,179,200,0.2);
                    background: #122033;
                    color: var(--text);
                    font-size: 1rem;
                }
                button {
                    cursor: pointer;
                    background: linear-gradient(135deg, var(--accent), var(--accent2));
                    color: #04111e;
                    font-weight: 800;
                    border: none;
                }
                button.secondary {
                    background: linear-gradient(135deg, var(--warning), #f59e0b);
                    color: #1c1200;
                }
                button.danger {
                    background: linear-gradient(135deg, var(--danger), #b91c1c);
                    color: white;
                }
                .two-col {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 12px;
                }
                .status {
                    min-height: 22px;
                    color: var(--muted);
                    font-weight: 700;
                }
                .cart {
                    display: grid;
                    gap: 10px;
                    margin-top: 12px;
                }
                .item {
                    display: grid;
                    grid-template-columns: 48px 1fr auto;
                    gap: 12px;
                    padding: 10px 12px;
                    background: rgba(18,32,51,0.9);
                    border: 1px solid rgba(159,179,200,0.1);
                    border-radius: 10px;
                    align-items: center;
                }
                .thumb {
                    width: 48px;
                    height: 48px;
                    border-radius: 9px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #1e293b, #334155);
                    font-weight: 700;
                    color: var(--warning);
                }
                .muted { color: var(--muted); }
                .product-list {
                    display: grid;
                    gap: 8px;
                    margin-top: 12px;
                }
                .badge {
                    display: inline-block;
                    background: rgba(45,212,191,0.1);
                    border: 1px solid rgba(45,212,191,0.3);
                    padding: 6px 10px;
                    border-radius: 999px;
                    color: var(--accent);
                    font-size: 0.8rem;
                    font-weight: 700;
                }
                .hidden { display: none; }
                .camera-frame {
                    position: relative;
                    display: inline-block;
                    width: 100%;
                }
                .camera-guide {
                    position: absolute;
                    top: 12%;
                    left: 20%;
                    width: 60%;
                    height: 76%;
                    border: 3px dashed rgba(45,212,191,0.85);
                    border-radius: 12px;
                    pointer-events: none;
                    box-shadow: 0 0 0 999px rgba(0,0,0,0.35);
                }
                .camera-guide::after {
                    content: 'Center the product in this frame';
                    position: absolute;
                    bottom: -28px;
                    left: 50%;
                    transform: translateX(-50%);
                    white-space: nowrap;
                    font-size: 0.8rem;
                    color: var(--accent);
                }
                @media (max-width: 900px) {
                    .panel-grid, .two-col { grid-template-columns: 1fr; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div>
                        <h1>Retail POS</h1>
                        <div class="badge">Learning + Checkout</div>
                    </div>
                </div>

                <div class="tabs">
                    <button class="tab active" data-tab="learning" type="button">Learning</button>
                    <button class="tab" data-tab="catalog" type="button">Catalog</button>
                    <button class="tab" data-tab="checkout" type="button">Checkout</button>
                </div>

                <div id="learning" class="panel-grid">
                    <div class="panel">
                        <h2>Add product to learning set</h2>
                        <form id="catalog-form" class="stack">
                            <div class="two-col">
                                <input id="product-name" name="name" type="text" placeholder="Product name" required />
                                <input id="product-barcode" name="barcode" type="text" placeholder="Barcode" required />
                            </div>
                            <div class="two-col">
                                <input id="product-price" name="price" type="number" step="0.01" min="0" placeholder="Price" required />
                                <input id="product-image" name="file" type="file" accept="image/*" required />
                            </div>
                            <button type="submit">Save product</button>
                        </form>
                        <div id="learning-status" class="status">Ready to add a product.</div>
                    </div>

                    <div class="panel">
                        <h3>Training</h3>
                        <div class="stack">
                            <button id="train-model" type="button" class="secondary">Train model</button>
                            <div id="training-status" class="status">Model not trained yet.</div>
                        </div>
                    </div>

                    <div class="panel" style="grid-column: 1 / -1;">
                        <h3>Or capture with your webcam</h3>
                        <p class="muted">Uses the barcode / name / price fields above. Re-using an existing barcode adds these photos to that product instead of replacing it — this is how you add more training images to a product you already onboarded.</p>
                        <div class="camera-frame" style="max-width:480px;">
                            <video id="learning-video" autoplay playsinline muted style="width:100%;border-radius:10px;background:#000;display:block;"></video>
                            <div class="camera-guide"></div>
                        </div>
                        <div class="two-col" style="margin-top:12px;">
                            <button id="learning-camera-toggle" type="button">Start Camera</button>
                            <button id="learning-capture-burst" type="button" class="secondary">Capture 12 Photos</button>
                        </div>
                        <div id="learning-camera-status" class="status"></div>
                    </div>

                    <div class="panel" style="grid-column: 1 / -1;">
                        <h3>Or bulk-import existing photos</h3>
                        <p class="muted">Select several photos already on disk, named <code>&lt;barcode&gt;.&lt;number&gt;.jpg</code> (e.g. <code>7290000504278.1.jpg</code>, <code>7290000504278.2.jpg</code>). The barcode is read from the filename, so photos for several products can be selected together. A barcode not yet in the catalog is created with a placeholder name/price — re-save it once through the form above to fill those in.</p>
                        <div class="stack">
                            <input id="bulk-import-files" type="file" accept="image/*" multiple />
                            <button id="bulk-import-btn" type="button" class="secondary">Import selected photos</button>
                        </div>
                        <div id="bulk-import-status" class="status"></div>
                    </div>
                </div>

                <div id="catalog" class="panel hidden">
                    <div class="panel">
                        <h2>Product catalog</h2>
                        <div id="catalog-list" class="product-list"></div>
                    </div>
                </div>

                <div id="checkout" class="panel hidden">
                    <div class="panel-grid">
                        <div class="panel">
                            <h2>Checkout scan</h2>
                            <div class="camera-frame">
                                <video id="checkout-video" autoplay playsinline muted style="width:100%;border-radius:10px;background:#000;display:block;"></video>
                                <div class="camera-guide"></div>
                            </div>
                            <div class="stack" style="margin-top:12px;">
                                <button id="checkout-camera-toggle" type="button">Start Camera (live scan)</button>
                                <input id="barcode-input" type="text" placeholder="Barcode or product name" />
                                <button id="add-item-btn" type="button">Add item</button>
                                <form id="upload-form" class="stack">
                                    <input id="file-input" type="file" accept="image/*" />
                                    <button type="submit">Analyze shelf image</button>
                                </form>
                                <div id="checkout-status" class="status">Ready for scan.</div>
                                <div id="suggestions" class="product-list"></div>
                            </div>
                        </div>

                        <div class="panel">
                            <h2>Cart</h2>
                            <div id="cart" class="cart"></div>
                            <div class="status" id="totals">Total: ₪0.00</div>
                            <div class="two-col">
                                <button id="complete-order-btn" type="button">Complete Order</button>
                                <button id="clear-cart-btn" type="button" class="danger">Clear Cart</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                const tabs = document.querySelectorAll('.tab');
                const sections = { learning: document.getElementById('learning'), catalog: document.getElementById('catalog'), checkout: document.getElementById('checkout') };
                let trainingPollHandle = null;

                tabs.forEach((tab) => {
                    tab.addEventListener('click', () => {
                        tabs.forEach(t => t.classList.toggle('active', t === tab));
                        Object.entries(sections).forEach(([name, el]) => {
                            el.classList.toggle('hidden', name !== tab.dataset.tab);
                        });
                    });
                });

                // ---- Shared webcam helpers (used by both Learning and Checkout) ----
                async function startCamera(videoEl, statusEl) {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                        videoEl.srcObject = stream;
                        await videoEl.play();
                        return stream;
                    } catch (err) {
                        statusEl.textContent = 'Camera access failed: ' + err.message;
                        return null;
                    }
                }

                function stopCamera(stream) {
                    if (stream) stream.getTracks().forEach(t => t.stop());
                }

                function captureFrameBlob(videoEl) {
                    const canvas = document.createElement('canvas');
                    canvas.width = videoEl.videoWidth;
                    canvas.height = videoEl.videoHeight;
                    canvas.getContext('2d').drawImage(videoEl, 0, 0);
                    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
                }

                function renderCatalogList() {
                    fetch('/catalog')
                        .then(r => r.json())
                        .then(data => {
                            const list = document.getElementById('catalog-list');
                            const items = data.products || {};
                            const rows = Object.values(items);
                            list.innerHTML = rows.length ? rows.map(item => `
                                <div class="item">
                                    <div class="thumb">${item.front_image_url ? `<img src="${item.front_image_url}" alt="${item.name}" style="width:100%;height:100%;object-fit:cover;border-radius:9px;">` : '#'}</div>
                                    <div>
                                        <strong>${item.name || item.barcode}</strong><br>
                                        <span class="muted">${item.barcode} &middot; ${item.image_count || 0} training images</span>
                                    </div>
                                    <div><strong>₪${Number(item.price || 0).toFixed(2)}</strong></div>
                                </div>
                            `).join('') : '<div class="muted">No products added yet.</div>';
                        })
                        .catch(() => {
                            document.getElementById('catalog-list').innerHTML = '<div class="muted">Catalog is empty.</div>';
                        });
                }

                // ---- Real, server-side cart (backed by pos_app's ShoppingCart / orders) ----
                function renderCart(cartData) {
                    document.getElementById('totals').textContent = 'Total: ₪' + Number(cartData.total || 0).toFixed(2);
                    const cartEl = document.getElementById('cart');
                    const items = cartData.items || [];
                    cartEl.innerHTML = items.length ? items.map(item => `
                        <div class="item">
                            <div class="thumb">${(item.name || item.barcode)[0]}</div>
                            <div>
                                <strong>${item.name}</strong><br>
                                <span class="muted">${item.barcode} &middot; Qty ${item.quantity}</span>
                            </div>
                            <div><strong>₪${Number(item.subtotal || 0).toFixed(2)}</strong></div>
                        </div>
                    `).join('') : '<div class="muted">No items in cart.</div>';
                }

                async function refreshCart() {
                    const response = await fetch('/checkout/cart');
                    renderCart(await response.json());
                }

                function renderSuggestions(detections) {
                    const box = document.getElementById('suggestions');
                    const weak = detections.filter(d => !d.is_confident);
                    if (!weak.length) {
                        box.innerHTML = '';
                        return;
                    }
                    box.innerHTML = '<div class="muted">Possible matches — click to confirm:</div>' + weak.map((d, i) => `
                        <div class="item">
                            <div class="thumb">?</div>
                            <div><strong>${d.name}</strong><br><span class="muted">${Math.round(d.confidence * 100)}% confidence</span></div>
                            <button type="button" data-idx="${i}" class="confirm-suggestion">Confirm</button>
                        </div>
                    `).join('');
                    box.querySelectorAll('.confirm-suggestion').forEach((btn) => {
                        btn.addEventListener('click', async () => {
                            const d = weak[Number(btn.dataset.idx)];
                            await fetch('/checkout/add', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ barcode: d.barcode, record_training: true, crop_path: d.crop_path }),
                            });
                            box.innerHTML = '';
                            await refreshCart();
                        });
                    });
                }

                document.getElementById('catalog-form').addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const form = event.target;
                    const formData = new FormData(form);
                    const response = await fetch('/catalog', { method: 'POST', body: formData });
                    const data = await response.json();
                    document.getElementById('learning-status').textContent = response.ok ? `Saved ${data.name} (${data.barcode})` : (data.detail || 'Failed to save product');
                    form.reset();
                    renderCatalogList();
                });

                // ---- Learning tab: webcam capture (new product OR more photos for an existing one) ----
                let learningStream = null;

                document.getElementById('learning-camera-toggle').addEventListener('click', async () => {
                    const btn = document.getElementById('learning-camera-toggle');
                    const video = document.getElementById('learning-video');
                    const statusEl = document.getElementById('learning-camera-status');
                    if (learningStream) {
                        stopCamera(learningStream);
                        learningStream = null;
                        btn.textContent = 'Start Camera';
                        statusEl.textContent = '';
                    } else {
                        learningStream = await startCamera(video, statusEl);
                        if (learningStream) btn.textContent = 'Stop Camera';
                    }
                });

                document.getElementById('learning-capture-burst').addEventListener('click', async () => {
                    const statusEl = document.getElementById('learning-camera-status');
                    const barcode = document.getElementById('product-barcode').value.trim();
                    const name = document.getElementById('product-name').value.trim();
                    const price = document.getElementById('product-price').value;
                    const video = document.getElementById('learning-video');

                    if (!learningStream) { statusEl.textContent = 'Start the camera first.'; return; }
                    if (!barcode) { statusEl.textContent = 'Enter a barcode first (existing or new).'; return; }
                    if (!name || !price) { statusEl.textContent = 'Enter a name and price first.'; return; }

                    const totalShots = 12;
                    for (let i = 0; i < totalShots; i++) {
                        statusEl.textContent = `Capturing ${i + 1}/${totalShots}… rotate the product slowly.`;
                        const blob = await captureFrameBlob(video);
                        const formData = new FormData();
                        formData.append('name', name);
                        formData.append('barcode', barcode);
                        formData.append('price', price);
                        formData.append('is_front', i === 0 ? 'true' : 'false');
                        formData.append('file', blob, `capture_${i}.jpg`);
                        await fetch('/catalog', { method: 'POST', body: formData });
                        await new Promise((r) => setTimeout(r, 350));
                    }
                    statusEl.textContent = `Captured ${totalShots} photos for barcode ${barcode}.`;
                    renderCatalogList();
                });

                document.getElementById('bulk-import-btn').addEventListener('click', async () => {
                    const statusEl = document.getElementById('bulk-import-status');
                    const input = document.getElementById('bulk-import-files');
                    const fileList = input.files;
                    if (!fileList || fileList.length === 0) {
                        statusEl.textContent = 'Choose one or more photos first.';
                        return;
                    }

                    const formData = new FormData();
                    for (const f of fileList) formData.append('files', f, f.name);

                    statusEl.textContent = `Uploading ${fileList.length} photo(s)…`;
                    const response = await fetch('/catalog/bulk-import', { method: 'POST', body: formData });
                    const data = await response.json();

                    const okCount = data.results.filter(r => r.status === 'ok').length;
                    const errors = data.results.filter(r => r.status === 'error');
                    const perProduct = Object.entries(data.products_touched || {}).map(([bc, count]) => `${bc}: ${count} images`).join(', ');
                    let summary = `Imported ${okCount}/${fileList.length} photo(s). ${perProduct ? 'Now: ' + perProduct + '.' : ''}`;
                    if (errors.length) {
                        summary += ` Skipped ${errors.length}: ` + errors.map(e => `${e.filename} (${e.message})`).join('; ');
                    }
                    statusEl.textContent = summary;
                    input.value = '';
                    renderCatalogList();
                });

                async function pollTrainingStatus() {
                    const response = await fetch('/train-model/status');
                    const data = await response.json();
                    const statusEl = document.getElementById('training-status');
                    if (data.status === 'running') {
                        statusEl.textContent = 'Training in progress… this can take several minutes (dataset prep + fine-tuning).';
                        trainingPollHandle = setTimeout(pollTrainingStatus, 5000);
                    } else if (data.status === 'done') {
                        statusEl.textContent = 'Training complete — the checkout model will reload automatically on next use.';
                        renderCatalogList();
                    } else if (data.status === 'failed') {
                        statusEl.textContent = 'Training failed (see training_log.txt on the server).';
                    } else {
                        statusEl.textContent = 'Model not trained yet.';
                    }
                }

                document.getElementById('train-model').addEventListener('click', async () => {
                    if (trainingPollHandle) clearTimeout(trainingPollHandle);
                    document.getElementById('training-status').textContent = 'Starting training…';
                    const response = await fetch('/train-model', { method: 'POST' });
                    const data = await response.json();
                    if (!response.ok) {
                        document.getElementById('training-status').textContent = data.detail || 'Training failed to start.';
                        return;
                    }
                    pollTrainingStatus();
                });

                document.getElementById('add-item-btn').addEventListener('click', async () => {
                    const barcode = document.getElementById('barcode-input').value.trim();
                    if (!barcode) return;
                    document.getElementById('barcode-input').value = '';
                    await fetch('/checkout/add', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ barcode, record_training: false }),
                    });
                    await refreshCart();
                });

                document.getElementById('complete-order-btn').addEventListener('click', async () => {
                    const response = await fetch('/checkout/complete', { method: 'POST' });
                    const data = await response.json();
                    document.getElementById('checkout-status').textContent = response.ok
                        ? `Order complete — total ₪${Number(data.total).toFixed(2)} (receipt ${data.receipt})`
                        : (data.detail || 'Could not complete order.');
                    document.getElementById('suggestions').innerHTML = '';
                    await refreshCart();
                });

                document.getElementById('clear-cart-btn').addEventListener('click', async () => {
                    await fetch('/checkout/clear', { method: 'POST' });
                    document.getElementById('suggestions').innerHTML = '';
                    await refreshCart();
                });

                // ---- Checkout tab: live webcam scanning ----
                let checkoutStream = null;
                let checkoutScanHandle = null;
                const liveScanCooldown = {}; // barcode -> last-added timestamp (client-side duplicate-scan guard)

                async function handleDetections(detections, options) {
                    const useCooldown = !!(options && options.useCooldown);
                    const now = Date.now();
                    const confident = detections.filter((d) => d.is_confident);
                    let addedCount = 0;
                    for (const d of confident) {
                        if (useCooldown) {
                            const last = liveScanCooldown[d.barcode] || 0;
                            if (now - last < 3000) continue; // mirrors the desktop app's 3s cooldown
                            liveScanCooldown[d.barcode] = now;
                        }
                        await fetch('/checkout/add', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ barcode: d.barcode, record_training: false }),
                        });
                        addedCount++;
                    }
                    if (addedCount) await refreshCart();
                    renderSuggestions(detections);
                    return addedCount;
                }

                document.getElementById('checkout-camera-toggle').addEventListener('click', async () => {
                    const btn = document.getElementById('checkout-camera-toggle');
                    const video = document.getElementById('checkout-video');
                    const statusEl = document.getElementById('checkout-status');
                    if (checkoutStream) {
                        stopCamera(checkoutStream);
                        checkoutStream = null;
                        clearInterval(checkoutScanHandle);
                        checkoutScanHandle = null;
                        btn.textContent = 'Start Camera (live scan)';
                        statusEl.textContent = 'Live scan stopped.';
                    } else {
                        checkoutStream = await startCamera(video, statusEl);
                        if (checkoutStream) {
                            btn.textContent = 'Stop Camera';
                            statusEl.textContent = 'Live scan running…';
                            checkoutScanHandle = setInterval(scanLiveFrame, 1800);
                        }
                    }
                });

                async function scanLiveFrame() {
                    const video = document.getElementById('checkout-video');
                    if (!video.videoWidth) return;
                    const blob = await captureFrameBlob(video);
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');
                    const response = await fetch('/checkout/detect', { method: 'POST', body: formData });
                    if (!response.ok) return;
                    const data = await response.json();
                    const detections = data.detections || [];
                    const added = await handleDetections(detections, { useCooldown: true });
                    const statusEl = document.getElementById('checkout-status');
                    if (added) {
                        statusEl.textContent = `Live scan: added ${added} item(s) to the cart.`;
                    } else if (detections.length) {
                        statusEl.textContent = 'Live scan: possible match — see suggestions below.';
                    } else {
                        statusEl.textContent = 'Live scan running… nothing recognized yet.';
                    }
                }

                document.getElementById('upload-form').addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const file = document.getElementById('file-input').files[0];
                    if (!file) {
                        document.getElementById('checkout-status').textContent = 'Choose an image first.';
                        return;
                    }
                    const formData = new FormData();
                    formData.append('file', file);
                    document.getElementById('checkout-status').textContent = 'Analyzing photo…';
                    const response = await fetch('/checkout/detect', { method: 'POST', body: formData });
                    const data = await response.json();
                    if (!response.ok) {
                        document.getElementById('checkout-status').textContent = data.detail || 'Detection failed';
                        return;
                    }
                    const detections = data.detections || [];
                    if (!detections.length) {
                        document.getElementById('checkout-status').textContent = 'No products recognized in that photo.';
                        document.getElementById('suggestions').innerHTML = '';
                        return;
                    }

                    const added = await handleDetections(detections, { useCooldown: false });
                    document.getElementById('checkout-status').textContent = added
                        ? `Added ${added} confidently-recognized product(s) to the cart.`
                        : 'No confident match — check the suggestions below.';
                });

                renderCatalogList();
                refreshCart();
            </script>
        </body>
        </html>
        """,
        status_code=200,
    )


@app.get("/catalog")
def get_catalog_endpoint() -> dict:
    """Return the current product catalog for the learning UI, with browsable thumbnail URLs."""
    catalog = get_catalog()
    products: dict[str, dict] = {}
    for barcode, product in catalog.items():
        entry = dict(product)
        front_path = product.get("front_image_path")
        entry["front_image_url"] = None
        if front_path:
            try:
                relative = Path(front_path).resolve().relative_to(DATASET_IMAGES_DIR.resolve())
                entry["front_image_url"] = f"/product-images/{relative.as_posix()}"
            except (ValueError, OSError):
                pass
        products[barcode] = entry
    return {"products": products}


@app.post("/catalog")
async def add_catalog_item(
    name: str = Form(...),
    barcode: str = Form(...),
    price: float = Form(...),
    file: UploadFile | None = File(None),
    is_front: bool = Form(True),
) -> dict:
    """Save one training photo for a product — new or already onboarded.

    Submitting an existing barcode again *adds* this photo to that
    product's training set rather than replacing it (matching the desktop
    onboarding screen). Each photo gets its own sequential filename
    (`img_N.<ext>`, continuing from whatever's already on disk); pass
    `is_front=true` (the default, for the single-photo "Save product" form)
    to additionally save it as the catalog's representative `front.<ext>`
    thumbnail.
    """
    if not name or not barcode:
        raise HTTPException(status_code=400, detail="Name and barcode are required.")
    if not math.isfinite(price) or price < 0:
        raise HTTPException(status_code=400, detail="Price must be a non-negative number.")
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="A product image is required.")

    target_dir = DATASET_IMAGES_DIR / barcode
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Product image must be JPG, PNG, or WebP.")

    image_bytes = await file.read()
    await file.close()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

    index = next_capture_index(barcode)
    image_path = target_dir / f"img_{index}{suffix}"
    image_path.write_bytes(image_bytes)

    front_image_path = None
    if is_front:
        front_image_path = target_dir / f"front{suffix}"
        front_image_path.write_bytes(image_bytes)

    register_product(
        barcode=barcode,
        name=name,
        price=float(price),
        image_paths=[image_path],
        front_image_path=front_image_path,
    )
    product = get_product(barcode)
    return {
        "status": "ok",
        "name": name,
        "barcode": barcode,
        "price": float(price),
        "image_count": product["image_count"] if product else 1,
    }


_ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@app.post("/catalog/bulk-import")
async def bulk_import_catalog_photos(files: list[UploadFile] = File(...)) -> dict:
    """Import already-taken photos whose filenames encode their product.

    Expects each filename in the form `<barcode>.<anything>.<ext>` (e.g.
    `7290000504278.1.jpg`, `7290000504278.2.jpg`) — the barcode is read from
    the part before the first dot, and the middle part is just there to keep
    filenames unique on disk; it isn't otherwise used since each photo gets
    its own sequential `img_N` name via `next_capture_index`. This lets a
    whole folder of pre-taken photos for one or more products be dropped in
    at once, instead of capturing through the live camera.

    A barcode that isn't in the catalog yet is created with a placeholder
    name (the barcode itself) and price 0 — re-submit it through the normal
    "Save product" form (name + barcode + price + one photo) to fill those
    in; `register_product` merges rather than overwrites, so that won't
    duplicate the photos already imported here.
    """
    results = []
    touched: dict[str, int] = {}

    for upload in files:
        filename = upload.filename or ""
        parts = filename.split(".")
        if len(parts) < 3:
            results.append({"filename": filename, "status": "error", "message": "Expected <barcode>.<number>.<ext>"})
            await upload.close()
            continue

        barcode = parts[0].strip()
        suffix = f".{parts[-1].lower()}"
        if not barcode:
            results.append({"filename": filename, "status": "error", "message": "Missing barcode in filename"})
            await upload.close()
            continue
        if suffix not in _ALLOWED_IMAGE_SUFFIXES:
            results.append({"filename": filename, "status": "error", "message": f"Unsupported extension {suffix}"})
            await upload.close()
            continue

        image_bytes = await upload.read()
        await upload.close()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            results.append({"filename": filename, "status": "error", "message": "Not a readable image"})
            continue

        target_dir = DATASET_IMAGES_DIR / barcode
        target_dir.mkdir(parents=True, exist_ok=True)
        index = next_capture_index(barcode)
        image_path = target_dir / f"img_{index}{suffix}"
        image_path.write_bytes(image_bytes)

        register_product(barcode=barcode, name="", price=0.0, image_paths=[image_path])
        product = get_product(barcode)
        touched[barcode] = product["image_count"] if product else touched.get(barcode, 0) + 1
        results.append({"filename": filename, "status": "ok", "barcode": barcode, "saved_as": image_path.name})

    return {"results": results, "products_touched": touched}


@app.post("/checkout/detect")
async def checkout_detect(file: UploadFile = File(...)) -> dict:
    """Run the fine-tuned checkout model on an uploaded photo.

    Read-only: returns candidate matches (confident and weak) without
    touching the cart. The frontend decides whether to auto-add a confident
    match or offer a weak one as a click-to-confirm suggestion — mirroring
    the desktop Checkout screen's suggestion strip.
    """
    image_bytes = await file.read()
    await file.close()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

    try:
        live_detector = _get_checkout_detector()
    except DetectorInitError as exc:
        raise HTTPException(status_code=503, detail=f"Checkout model failed to load: {exc}") from exc

    h, w = image.shape[:2]
    candidates = []
    for det in live_detector.infer(image):
        if det.confidence < SUGGESTION_MIN_CONFIDENCE:
            continue
        x1, y1, x2, y2 = det.bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else image

        crop_name = f"web_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / "crops" / crop_name), crop)

        product = get_product(det.label)
        candidates.append(
            {
                "barcode": det.label,
                "name": product["name"] if product else det.label,
                "price": product["price"] if product else 0.0,
                "confidence": round(det.confidence, 4),
                "is_confident": det.confidence >= POS_CONFIDENCE_THRESHOLD,
                "crop_path": f"/outputs/crops/{crop_name}",
            }
        )

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return {"detections": candidates[: MAX_SUGGESTIONS + 2], "using_custom_model": live_detector.using_custom_model}


@app.post("/checkout/add")
async def checkout_add(payload: CheckoutAddRequest) -> dict:
    """Add one unit of `barcode` to the server-side checkout cart.

    Set `record_training=true` (with `crop_path` from a prior
    `/checkout/detect` call) when this add is a *human-confirmed* match —
    e.g. a clicked suggestion — so the photo folds back into that
    product's training set. Automatic high-confidence adds and plain typed
    barcodes should leave this false, matching the desktop app's policy of
    only learning from verified confirmations.
    """
    barcode = payload.barcode.strip()
    if not barcode:
        raise HTTPException(status_code=400, detail="Barcode is required.")

    product = get_product(barcode)
    name = product["name"] if product else barcode
    price = product["price"] if product else 0.0

    checkout_cart.mark_scanned(barcode)
    checkout_cart.add_or_increment(barcode, name, price)

    if payload.record_training and payload.crop_path:
        crop_file = OUTPUT_DIR / "crops" / Path(payload.crop_path).name
        if crop_file.exists():
            crop_image = cv2.imread(str(crop_file))
            if crop_image is not None:
                record_checkout_training_sample(barcode, crop_image)

    return _cart_payload()


@app.get("/checkout/cart")
async def checkout_get_cart() -> dict:
    return _cart_payload()


@app.post("/checkout/clear")
async def checkout_clear() -> dict:
    checkout_cart.clear()
    return _cart_payload()


@app.post("/checkout/complete")
async def checkout_complete() -> dict:
    """Finalize the sale: save a JSON receipt (like the desktop app) and reset the cart."""
    if not checkout_cart.items:
        raise HTTPException(status_code=400, detail="Cart is empty.")

    items_payload = [
        {
            "barcode": item.barcode,
            "name": item.name,
            "quantity": item.quantity,
            "price": item.price,
            "subtotal": round(item.price * item.quantity, 2),
        }
        for item in checkout_cart.items.values()
    ]
    total = checkout_cart.total()
    receipt_path = save_order(items_payload, total)
    checkout_cart.clear()
    return {"status": "ok", "receipt": receipt_path.name, "total": round(total, 2), "items": items_payload}


@app.post("/train-model")
async def train_model() -> dict:
    """Start the fine-tuning job for the product catalog in the background.

    Runs as a separate OS process (`subprocess.Popen`, not `.run()`) so the
    FastAPI event loop — and every other endpoint — stays responsive while
    training runs; dataset prep alone can take many minutes. Poll
    `/train-model/status` for progress.
    """
    global training_process

    catalog = get_catalog()
    usable_images = sum(
        1
        for product in catalog.values()
        for image_path in product.get("image_paths", [])
        if Path(image_path).exists()
    )
    if not catalog:
        raise HTTPException(status_code=400, detail="Add at least one product before training.")
    if not usable_images:
        raise HTTPException(status_code=400, detail="No readable product images are available for training.")
    if training_process is not None and training_process.poll() is None:
        raise HTTPException(status_code=409, detail="Training is already in progress.")

    try:
        log_file = training_log_path.open("w", encoding="utf-8")
        training_process = subprocess.Popen(
            [sys.executable, "-m", "pos_app.train_yolo", "--epochs", "30", "--base-model", "yolo11n.pt"],
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Training failed to start: {exc}") from exc

    return {"status": "started"}


@app.get("/train-model/status")
async def train_model_status() -> dict:
    """Poll the background training job started by `POST /train-model`."""
    global training_process, checkout_detector

    if training_process is None:
        return {"status": "idle"}
    if training_process.poll() is None:
        return {"status": "running"}

    returncode = training_process.returncode
    training_process = None  # consume the terminal state so later polls report "idle"
    checkout_detector = None  # force a reload so the newly fine-tuned weights take effect immediately
    if returncode == 0:
        return {"status": "done"}
    return {"status": "failed", "returncode": returncode}


@app.on_event("startup")
def load_model() -> None:
    """Load the YOLO model once at startup so `/detect` requests stay fast."""
    global detector
    try:
        detector = RetailProductDetector()
        logger.info("Model loaded successfully: %s", detector.health())
    except DetectorInitError as exc:
        logger.error("Model failed to load at startup: %s", exc)
        detector = None


@app.get("/health")
def health() -> dict:
    """Return GPU/CPU status and model load state."""
    if detector is None:
        return JSONResponse(
            status_code=503,
            content={"model_loaded": False, "error": "Model is not loaded. Check server logs."},
        )
    return detector.health()


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    """Accept an uploaded image, run detection, and return the structured JSON payload."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded; check /health for details.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {exc}") from exc
    finally:
        await file.close()

    try:
        stem = Path(file.filename or tmp_path.name).stem
        payload = detector.detect_image(tmp_path, output_stem=stem)
    except DetectorInferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return payload
