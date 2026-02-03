/**
 * E-commerce Namespace API
 *
 * Provides endpoints for e-commerce integration including
 * product catalogs, orders, and customer management.
 */

import type { AragoraClient } from '../client';

/** Product in the catalog */
export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  currency: string;
  category: string;
  sku: string;
  in_stock: boolean;
  image_url?: string;
  created_at: string;
}

/** E-commerce order */
export interface Order {
  id: string;
  customer_id: string;
  status: 'pending' | 'processing' | 'shipped' | 'delivered' | 'cancelled';
  items: OrderItem[];
  total: number;
  currency: string;
  created_at: string;
  updated_at: string;
}

/** Order line item */
export interface OrderItem {
  product_id: string;
  product_name: string;
  quantity: number;
  unit_price: number;
  total: number;
}

/** E-commerce analytics */
export interface EcommerceAnalytics {
  total_revenue: number;
  total_orders: number;
  average_order_value: number;
  top_products: Array<{ product_id: string; name: string; sold: number }>;
  period: string;
}

/**
 * E-commerce namespace for product and order management.
 *
 * @example
 * ```typescript
 * const products = await client.ecommerce.listProducts();
 * const orders = await client.ecommerce.listOrders({ status: 'pending' });
 * ```
 */
export class EcommerceNamespace {
  constructor(private client: AragoraClient) {}

  /** List products. */
  async listProducts(options?: {
    category?: string;
    limit?: number;
    offset?: number;
  }): Promise<Product[]> {
    const response = await this.client.request<{ products: Product[] }>(
      'GET',
      '/api/v1/ecommerce/products',
      { params: options }
    );
    return response.products;
  }

  /** Get a product by ID. */
  async getProduct(productId: string): Promise<Product> {
    return this.client.request<Product>(
      'GET',
      `/api/v1/ecommerce/products/${encodeURIComponent(productId)}`
    );
  }

  /** List orders. */
  async listOrders(options?: {
    status?: string;
    customer_id?: string;
    limit?: number;
  }): Promise<Order[]> {
    const response = await this.client.request<{ orders: Order[] }>(
      'GET',
      '/api/v1/ecommerce/orders',
      { params: options }
    );
    return response.orders;
  }

  /** Get an order by ID. */
  async getOrder(orderId: string): Promise<Order> {
    return this.client.request<Order>(
      'GET',
      `/api/v1/ecommerce/orders/${encodeURIComponent(orderId)}`
    );
  }

  /** Get e-commerce analytics. */
  async getAnalytics(options?: { period?: string }): Promise<EcommerceAnalytics> {
    return this.client.request<EcommerceAnalytics>(
      'GET',
      '/api/v1/ecommerce/analytics',
      { params: options }
    );
  }
}
